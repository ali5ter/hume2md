#!/usr/bin/env python3
"""hume2md — Convert a Hume Body Pod progress-report PNG into labeled Markdown.

Hume Health exports body-composition progress reports only as PNG screenshots,
which are awkward to read programmatically (and easy to misread by eye). This
tool runs on-device Apple Vision OCR (via the ``ocrmac`` package) over the
image, clusters the recognized text into rows, binds each known Hume metric's
label to the previous/current values rendered on its card, and writes a
Markdown summary. Parsed values are checked against per-metric plausibility
ranges and cross-metric invariants before being trusted; failures are flagged
rather than silently reported as good data. The raw OCR text is always
appended so the output is useful even when parsing misses a field.

Author: Alister Lewis-Bowen <alister@lewis-bowen.org>
Version: 0.2.0
Date: 2026-08-23
License: MIT
Usage:
    ./hume2md.py REPORT.png [-o OUTPUT.md] [--raw] [--date YYYY-MM-DD]
                             [--csv OUTPUT.csv]
Dependencies:
    Python 3.10+, ocrmac (https://github.com/straussmaximilian/ocrmac), rich.
    Requires macOS (ocrmac wraps the Apple Vision framework).
Exit codes:
    0 = success
    1 = input/usage error
    2 = OCR/runtime error
    3 = one or more metrics failed validation (Markdown/CSV are still written)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

console = Console()
err_console = Console(stderr=True, style="red")

# Row clustering tolerance in normalized image coordinates (0-1). Tokens whose
# vertical centres fall within this band are treated as the same row.
ROW_Y_TOLERANCE = 0.015

# How many rows below a label row to search for its numeric values. Hume
# renders a card as label, then values, then a status band, so 3 covers that
# plus slack — but stops well short of the next unrelated card.
CARD_FORWARD_WINDOW = 3

# Max normalized x-distance from a label's own tokens that a numeric token may
# be bound to it. Keeps a neighbouring card sharing a row (the summary strip)
# or a nearby row from being absorbed into this metric.
CARD_MAX_X_SPAN = 0.4

# Word-boundary, mutually exclusive per metric. Order does not disambiguate
# overlap anymore (each pattern is anchored so it cannot match another
# metric's label) — it only affects the order metrics appear in the output.
METRIC_SPECS: list[tuple[str, re.Pattern[str], str | None]] = [
    ("Health Score", re.compile(r"\bhealth score\b"), None),
    ("Body Fat Mass", re.compile(r"\bbody fat mass\b"), "lb"),
    ("Subcutaneous Fat Mass", re.compile(r"\bsubcutaneous fat mass\b"), "lb"),
    ("Body Fat %", re.compile(r"\bbody fat\s*%"), "%"),
    ("Fat Free Mass", re.compile(r"\bfat[- ]free mass\b"), "lb"),
    ("Lean Mass", re.compile(r"\blean (?:mass|body mass)\b"), "lb"),
    ("Skeletal Muscle Mass", re.compile(r"\bskeletal muscle mass\b"), "lb"),
    ("Bone Mass", re.compile(r"\bbone mass\b"), "lb"),
    ("Visceral Fat Index", re.compile(r"\bvisceral fat(?: index)?\b"), None),
    ("Body Water %", re.compile(r"\bbody water\s*%"), "%"),
    ("Protein", re.compile(r"\bprotein\b"), "lb"),
    ("BMR", re.compile(r"\b(?:bmr|basal metabolic rate)\b"), "cal"),
    ("Metabolic Age", re.compile(r"\bmetabolic age\b"), "years"),
    ("Resting Heart Rate", re.compile(r"\bresting heart rate\b"), "bpm"),
    ("Weight", re.compile(r"\bweight\b"), "lb"),
    ("Body Cell Mass", re.compile(r"\bbody cell mass\b"), "lb"),
]

# Per-metric plausibility ranges (inclusive). Metrics with no entry are
# accepted as-is — no known bound to check them against.
METRIC_RANGES: dict[str, tuple[float, float]] = {
    "Weight": (80, 400),
    "Body Fat %": (3, 60),
    "Visceral Fat Index": (1, 30),
    "Metabolic Age": (18, 100),
    "BMR": (800, 3000),
}

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Token:
    """A single OCR text fragment with its normalized centre position."""

    text: str
    x: float
    y: float


@dataclass
class Metric:
    """A parsed Hume metric row."""

    label: str
    previous: str | None
    current: str | None
    unit: str | None = None
    unverified: bool = False


@dataclass
class Report:
    """Parsed report plus the raw OCR lines used to build it."""

    metrics: list[Metric] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Convert a Hume Body Pod progress-report PNG to Markdown.",
    )
    parser.add_argument("image", type=Path, help="Path to the Hume PNG export.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Markdown output path (default: alongside the image).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Emit only the raw OCR text (skip metric parsing).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Report date label (YYYY-MM-DD); inferred from filename if omitted.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "Also write metrics as CSV rows (date,metric,value,unit) for "
            "appending to a metrics log. Unverified metrics are omitted."
        ),
    )
    return parser.parse_args(argv)


def ocr_tokens(image: Path) -> list[Token]:
    """Run Apple Vision OCR over an image and return positioned tokens.

    Args:
        image: Path to the image file.

    Returns:
        A list of :class:`Token` with normalized centre coordinates (origin
        bottom-left, as returned by Vision).

    Raises:
        SystemExit: If ``ocrmac`` is unavailable or OCR fails.
    """
    try:
        from ocrmac import ocrmac
    except ImportError:
        err_console.print(
            "ocrmac is not installed. Run: pip install ocrmac rich "
            "(macOS only — it wraps the Apple Vision framework).",
        )
        raise SystemExit(2) from None

    try:
        annotations = ocrmac.OCR(
            str(image), recognition_level="accurate", language_preference=["en-US"]
        ).recognize()
    except Exception as exc:  # noqa: BLE001 - surface any Vision error cleanly
        err_console.print(f"OCR failed: {exc}")
        raise SystemExit(2) from exc

    tokens: list[Token] = []
    for text, _confidence, bbox in annotations:
        x, y, w, h = bbox
        tokens.append(Token(text=text.strip(), x=x + w / 2, y=y + h / 2))
    return tokens


def cluster_rows(tokens: list[Token]) -> list[list[Token]]:
    """Group tokens into rows by vertical proximity, top-to-bottom.

    Args:
        tokens: OCR tokens with normalized centres.

    Returns:
        Rows (each a list of tokens sorted left-to-right), ordered top-to-bottom.
    """
    rows: list[list[Token]] = []
    for token in sorted(tokens, key=lambda t: -t.y):  # Vision origin: top = high y
        for row in rows:
            if abs(row[0].y - token.y) <= ROW_Y_TOLERANCE:
                row.append(token)
                break
        else:
            rows.append([token])
    for row in rows:
        row.sort(key=lambda t: t.x)
    return rows


def _label_span(row: list[Token], pattern: re.Pattern[str]) -> tuple[int, int] | None:
    """Return the (start_index, end_index) of the tokens a label pattern matches.

    Joins the row into one string while tracking each token's character
    range, finds ``pattern``'s first match in that string, then maps the
    match's character span back to the tokens it covers. This lets a label
    that begins mid-row — two metrics sharing a horizontal summary-strip row
    — resolve to only its own tokens rather than the whole row's, since a
    naive expand-from-token-zero scan would keep absorbing leading tokens
    until the pattern happens to appear as a substring further along.

    Args:
        row: A clustered OCR row, sorted left-to-right.
        pattern: The metric's compiled label pattern.

    Returns:
        Indices of the first and last matched tokens, or ``None`` if the
        pattern does not match the row at all.
    """
    text = ""
    spans: list[tuple[int, int]] = []  # (start_char, end_char) per token
    for token in row:
        if text:
            text += " "
        start_char = len(text)
        text += token.text.lower()
        spans.append((start_char, len(text)))

    match = pattern.search(text)
    if match is None:
        return None
    start_idx = next(i for i, (_, e) in enumerate(spans) if e > match.start())
    end_idx = next(
        i for i in range(len(spans) - 1, -1, -1) if spans[i][0] < match.end()
    )
    return start_idx, end_idx


def _numbers_in_span(
    row: list[Token],
    row_idx: int,
    min_x: float,
    max_x: float,
    used: set[tuple[int, int]],
) -> list[tuple[int, Token]]:
    """Unclaimed numeric tokens in a row within [min_x, max_x], with index."""
    return [
        (i, t)
        for i, t in enumerate(row)
        if min_x <= t.x <= max_x
        and (row_idx, i) not in used
        and NUMBER_RE.search(t.text)
    ]


def _bind_metric(
    rows: list[list[Token]],
    pattern: re.Pattern[str],
    used: set[tuple[int, int]],
) -> tuple[str | None, str] | None:
    """Locate a metric's label and bind its previous/current values.

    Finds the first row whose text matches ``pattern`` at tokens not already
    claimed by another metric, then looks for numeric values on that same row
    and up to ``CARD_FORWARD_WINDOW`` rows below it — mirroring the Hume card
    layout, where values render directly beneath their label. Only unclaimed
    numeric tokens within ``CARD_MAX_X_SPAN`` of the label's own tokens are
    considered, so a neighbouring card sharing the row (the horizontal
    summary strip) or a nearby row is not absorbed into this metric. If no
    row in the window carries a usable numeric value, the metric is
    abandoned — the search does not continue past the window to some
    unrelated row later in the document.

    Tracking is per-token rather than per-row so two metrics whose labels sit
    side by side on one shared row (e.g. "Weight ... Metabolic Age ...") each
    claim only their own tokens instead of one consuming the whole row.

    Args:
        rows: Clustered OCR rows from :func:`cluster_rows`.
        pattern: The metric's compiled label pattern.
        used: (row_index, token_index) pairs already bound to another
            metric; mutated in place with the tokens this call claims.

    Returns:
        ``(previous, current)``, or ``None`` if the label was not found or no
        value bound to it.
    """
    for label_idx, label_row in enumerate(rows):
        row_text = " ".join(t.text for t in label_row).lower()
        if not pattern.search(row_text):
            continue

        span = _label_span(label_row, pattern)
        if span is None:
            return None
        start_idx, end_idx = span
        if any((label_idx, i) in used for i in range(start_idx, end_idx + 1)):
            continue  # This occurrence's tokens are already another metric's.

        label_x = label_row[start_idx].x
        max_x = label_row[end_idx].x + CARD_MAX_X_SPAN

        for offset in range(CARD_FORWARD_WINDOW + 1):
            value_idx = label_idx + offset
            if value_idx >= len(rows):
                continue
            candidates = _numbers_in_span(
                rows[value_idx], value_idx, label_x, max_x, used
            )
            if not candidates:
                continue
            candidates.sort(key=lambda pair: pair[1].x)
            previous = (
                NUMBER_RE.search(candidates[0][1].text).group()
                if len(candidates) >= 2
                else None
            )
            current = NUMBER_RE.search(candidates[-1][1].text).group()

            for i in range(start_idx, end_idx + 1):
                used.add((label_idx, i))
            used.add((value_idx, candidates[0][0]))
            used.add((value_idx, candidates[-1][0]))
            return previous, current
        return None  # Label found but no numeric row in window: abandon it.
    return None


def parse_metrics(rows: list[list[Token]]) -> list[Metric]:
    """Map known Hume labels to their previous/current values.

    Args:
        rows: Clustered OCR rows from :func:`cluster_rows`.

    Returns:
        Parsed metrics in canonical order, skipping any not found or whose
        values could not be bound within the card window.
    """
    metrics: list[Metric] = []
    used: set[tuple[int, int]] = set()
    for label, pattern, unit in METRIC_SPECS:
        bound = _bind_metric(rows, pattern, used)
        if bound is None:
            continue
        previous, current = bound
        metrics.append(Metric(label, previous, current, unit))
    return metrics


def validate_metrics(metrics: list[Metric]) -> list[str]:
    """Flag implausible metrics in place and return human-readable warnings.

    Applies per-metric plausibility ranges (:data:`METRIC_RANGES`) and
    cross-metric invariants that catch card-crossover errors a range check
    alone would miss (e.g. a metabolic age bound to a weight reading). Any
    metric that fails a check has ``Metric.unverified`` set so the renderer
    can flag it instead of reporting a wrong number as good data.

    Args:
        metrics: Parsed metrics, mutated in place.

    Returns:
        One warning message per failed check; empty if everything passed.
    """
    warnings: list[str] = []
    by_label = {m.label: m for m in metrics}

    def value(label: str) -> float | None:
        m = by_label.get(label)
        if m is None or m.current is None:
            return None
        return float(m.current)

    for m in metrics:
        bounds = METRIC_RANGES.get(m.label)
        if bounds is None or m.current is None:
            continue
        low, high = bounds
        current = float(m.current)
        if not (low <= current <= high):
            m.unverified = True
            warnings.append(
                f"{m.label} = {m.current} is outside the plausible range "
                f"[{low}, {high}]"
            )

    def flag_pair(a_label: str, b_label: str, message: str) -> None:
        by_label[a_label].unverified = True
        by_label[b_label].unverified = True
        warnings.append(message)

    cell, lean, weight = value("Body Cell Mass"), value("Lean Mass"), value("Weight")
    if cell is not None and lean is not None and not cell < lean:
        flag_pair(
            "Body Cell Mass",
            "Lean Mass",
            f"Body Cell Mass ({cell}) should be less than Lean Mass ({lean})",
        )
    if lean is not None and weight is not None and not lean < weight:
        flag_pair(
            "Lean Mass",
            "Weight",
            f"Lean Mass ({lean}) should be less than Weight ({weight})",
        )

    subq = value("Subcutaneous Fat Mass")
    body_fat_mass = value("Body Fat Mass")
    if subq is not None and body_fat_mass is not None and not subq <= body_fat_mass:
        flag_pair(
            "Subcutaneous Fat Mass",
            "Body Fat Mass",
            f"Subcutaneous Fat Mass ({subq}) should be <= Body Fat Mass "
            f"({body_fat_mass})",
        )

    return warnings


def infer_date(image: Path, explicit: str | None) -> str:
    """Determine a report date label from the flag or filename.

    Args:
        image: Source image path.
        explicit: A ``--date`` value, if supplied.

    Returns:
        An ISO date string when found, else an empty string.
    """
    if explicit:
        return explicit
    iso = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", image.stem)
    if iso:
        return "-".join(iso.groups())
    dmy = re.search(r"(\d{1,2})[_-]?([A-Za-z]{3})[_-]?(\d{4})", image.stem)
    if dmy:
        return f"{dmy.group(3)} {dmy.group(2)} {dmy.group(1)}"
    return ""


def render_markdown(report: Report, source: str, date: str, raw_only: bool) -> str:
    """Render the parsed report as Markdown.

    Args:
        report: Parsed metrics and raw OCR lines.
        source: Source image filename.
        date: Report date label (may be empty).
        raw_only: If true, emit only the raw OCR section.

    Returns:
        A Markdown document string.
    """
    lines = ["# Hume Body Pod — Progress Report", ""]
    lines.append(f"- Source: `{source}`")
    if date:
        lines.append(f"- Report date: {date}")
    lines.append("")

    if not raw_only and report.metrics:
        lines += ["| Metric | Previous | Current | Unit |", "|---|---|---|---|"]
        for m in report.metrics:
            current = "⚠️ unverified" if m.unverified else (m.current or "")
            lines.append(
                f"| {m.label} | {m.previous or ''} | {current} | {m.unit or ''} |"
            )
        lines += [
            "",
            "> Parsed from Apple Vision OCR by hume2md, binding each label to "
            "its own card. Rows flagged ⚠️ unverified failed a plausibility "
            "check — treat the source image as authoritative for those.",
            "",
        ]

    lines += ["## Raw OCR text", "", "```text"]
    lines += report.raw_lines
    lines += ["```", ""]
    return "\n".join(lines)


def write_csv(metrics: list[Metric], date: str, path: Path) -> int:
    """Write metrics as CSV rows for appending to an external metrics log.

    Rows flagged ``unverified`` by :func:`validate_metrics` are omitted so a
    plausibility failure cannot silently enter a downstream log — the
    Markdown output still records it, with a warning on stderr.

    Args:
        metrics: Parsed and validated metrics.
        date: Report date label, written into every row.
        path: Output CSV path.

    Returns:
        Number of rows written (excluding the header).
    """
    rows = [m for m in metrics if not m.unverified and m.current is not None]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "metric", "value", "unit"])
        for m in rows:
            writer.writerow([date, m.label, m.current, m.unit or ""])
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)

    if not args.image.is_file():
        err_console.print(f"No such file: {args.image}")
        return 1

    console.rule("[bold]hume2md[/bold]")
    console.print(f"[cyan]Reading[/cyan] {args.image}")

    tokens = ocr_tokens(args.image)
    rows = cluster_rows(tokens)
    metrics = [] if args.raw else parse_metrics(rows)
    warnings = [] if args.raw else validate_metrics(metrics)
    report = Report(
        metrics=metrics,
        raw_lines=[" ".join(t.text for t in row) for row in rows],
    )

    date = infer_date(args.image, args.date)
    markdown = render_markdown(report, args.image.name, date, args.raw)

    output = args.output or args.image.with_suffix(".md")
    output.write_text(markdown, encoding="utf-8")

    if args.csv:
        written = write_csv(metrics, date, args.csv)
        console.print(f"[green]✔[/green] Wrote {written} row(s) to {args.csv}")

    for warning in warnings:
        err_console.print(f"[yellow]![/yellow] {warning}")

    if report.metrics:
        console.print(f"[green]✔[/green] Parsed {len(report.metrics)} metrics")
    else:
        console.print("[yellow]![/yellow] No metrics parsed — raw OCR text only")
    console.print(f"[green]✔[/green] Wrote {output}")

    return 3 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
