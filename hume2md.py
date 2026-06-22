#!/usr/bin/env python3
"""hume2md — Convert a Hume Body Pod progress-report PNG into labeled Markdown.

Hume Health exports body-composition progress reports only as PNG screenshots,
which are awkward to read programmatically (and easy to misread by eye). This
tool runs on-device Apple Vision OCR (via the ``ocrmac`` package) over the
image, clusters the recognized text into rows, maps each known Hume metric
label to its previous/current values, and writes a Markdown summary. The raw
OCR text is always appended so the output is useful even when parsing misses a
field.

Author: Alister Lewis-Bowen <alister@lewis-bowen.org>
Version: 0.1.0
Date: 2026-06-22
License: MIT
Usage:
    ./hume2md.py REPORT.png [-o OUTPUT.md] [--raw] [--date YYYY-MM-DD]
Dependencies:
    Python 3.10+, ocrmac (https://github.com/straussmaximilian/ocrmac), rich.
    Requires macOS (ocrmac wraps the Apple Vision framework).
Exit codes:
    0 = success, 1 = input/usage error, 2 = OCR/runtime error
"""

from __future__ import annotations

import argparse
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

# Ordered most-specific-first so "Body Fat Mass" is matched before "Body Fat".
# Each entry: (canonical label, [lower-case match substrings], unit or None).
METRIC_SPECS: list[tuple[str, list[str], str | None]] = [
    ("Health Score", ["health score"], None),
    ("Body Fat Mass", ["body fat mass", "fat mass"], "lb"),
    ("Body Fat %", ["body fat"], "%"),
    ("Fat Free Mass", ["fat free", "fat-free"], "lb"),
    ("Lean Mass", ["lean mass", "lean body"], "lb"),
    ("Skeletal Muscle Mass", ["skeletal muscle"], "lb"),
    ("Bone Mass", ["bone mass", "skeletal mass"], "lb"),
    ("Visceral Fat", ["visceral"], None),
    ("Subcutaneous Fat", ["subcutaneous"], None),
    ("Body Water %", ["body water", "water"], "%"),
    ("Protein", ["protein"], "lb"),
    ("BMR", ["bmr", "basal", "metabolic rate"], "cal"),
    ("Metabolic Age", ["metabolic age"], "years"),
    ("Resting Heart Rate", ["heart rate", "resting"], "bpm"),
    ("Weight", ["weight"], "lb"),
]

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


def _row_numbers(row: list[Token]) -> list[str]:
    """Extract numeric values from a row, ordered left-to-right."""
    numbers: list[str] = []
    for token in row:
        match = NUMBER_RE.search(token.text)
        if match:
            numbers.append(match.group())
    return numbers


def parse_metrics(rows: list[list[Token]]) -> list[Metric]:
    """Map known Hume labels to their previous/current values.

    Assigns the left-most numeric token in a row as ``previous`` and the
    right-most as ``current`` (the Hume layout places the prior reading left of
    the latest). Verify against the source image; column order can vary.

    Args:
        rows: Clustered OCR rows from :func:`cluster_rows`.

    Returns:
        Parsed metrics in canonical order, skipping any not found.
    """
    metrics: list[Metric] = []
    used_rows: set[int] = set()
    for label, needles, unit in METRIC_SPECS:
        for idx, row in enumerate(rows):
            if idx in used_rows:
                continue
            row_text = " ".join(t.text for t in row).lower()
            if any(needle in row_text for needle in needles):
                numbers = _row_numbers(row)
                if not numbers:
                    continue
                previous = numbers[0] if len(numbers) >= 2 else None
                current = numbers[-1]
                metrics.append(Metric(label, previous, current, unit))
                used_rows.add(idx)
                break
    return metrics


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
            lines.append(
                f"| {m.label} | {m.previous or ''} | {m.current or ''} | {m.unit or ''} |"
            )
        lines += [
            "",
            "> Parsed from Apple Vision OCR by hume2md. Verify the "
            "previous/current columns against the source image.",
            "",
        ]

    lines += ["## Raw OCR text", "", "```text"]
    lines += report.raw_lines
    lines += ["```", ""]
    return "\n".join(lines)


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
    report = Report(
        metrics=[] if args.raw else parse_metrics(rows),
        raw_lines=[" ".join(t.text for t in row) for row in rows],
    )

    date = infer_date(args.image, args.date)
    markdown = render_markdown(report, args.image.name, date, args.raw)

    output = args.output or args.image.with_suffix(".md")
    output.write_text(markdown, encoding="utf-8")

    if report.metrics:
        console.print(f"[green]✔[/green] Parsed {len(report.metrics)} metrics")
    else:
        console.print("[yellow]![/yellow] No metrics parsed — raw OCR text only")
    console.print(f"[green]✔[/green] Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
