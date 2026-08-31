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
Version: 0.3.0
Date: 2026-08-31
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
import difflib
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

# Word-boundary, mutually exclusive per metric. Order does not disambiguate
# overlap anymore (each pattern is anchored so it cannot match another
# metric's label) — it only affects the order metrics appear in the output.
# Patterns accept the phrasing variants Hume has actually rendered across
# reports (e.g. "Body Fat Percentage" spelled out instead of "Body Fat %"),
# since exact substring matching cannot survive OCR-garbled labels at all —
# those fall through to the fuzzy-matching pass in _fuzzy_bind_metrics().
METRIC_SPECS: list[tuple[str, re.Pattern[str], str | None]] = [
    ("Health Score", re.compile(r"\bhealth score\b"), None),
    ("Body Fat Mass", re.compile(r"\bbody fat mass\b"), "lb"),
    ("Subcutaneous Fat Mass", re.compile(r"\bsubcutaneous fat mass\b"), "lb"),
    ("Body Fat %", re.compile(r"\bbody fat\s*%|\bbody fat percentage\b"), "%"),
    ("Fat Free Mass", re.compile(r"\bfat[- ]free mass\b"), "lb"),
    ("Lean Mass", re.compile(r"\blean (?:mass|body mass)\b"), "lb"),
    ("Skeletal Muscle Mass", re.compile(r"\bskeletal muscle mass\b"), "lb"),
    # Hume's card is labeled "Skeletal Mass" on screen, not "Bone Mass" — the
    # canonical name is kept for output clarity, but the pattern must match
    # what actually renders or this card is never claimed and its value
    # bleeds into Skeletal Muscle Mass's card-window search instead.
    ("Bone Mass", re.compile(r"\b(?:bone mass|skeletal mass)\b"), "lb"),
    ("Visceral Fat Index", re.compile(r"\bvisceral fat(?: index)?\b"), None),
    ("Body Water %", re.compile(r"\bbody water\b"), "%"),
    ("Protein", re.compile(r"\bprotein\b"), "lb"),
    ("BMR", re.compile(r"\b(?:bmr|basal metabolic rate)\b"), "cal"),
    ("Metabolic Age", re.compile(r"\bmetabolic age\b"), "years"),
    ("Resting Heart Rate", re.compile(r"\b(?:resting )?heart rate\b"), "bpm"),
    ("Weight", re.compile(r"\bweight\b"), "lb"),
    ("Body Cell Mass", re.compile(r"\bbody cell mass\b"), "lb"),
]

# Per-metric plausibility ranges (inclusive). Metrics with no entry are
# accepted as-is — no known bound to check them against.
METRIC_RANGES: dict[str, tuple[float, float]] = {
    "Weight": (80, 400),
    "Body Fat Mass": (5, 150),
    "Subcutaneous Fat Mass": (2, 100),
    "Body Fat %": (3, 60),
    "Lean Mass": (50, 250),
    "Skeletal Muscle Mass": (40, 150),
    "Bone Mass": (5, 50),
    "Visceral Fat Index": (1, 30),
    "Body Water %": (30, 80),
    "BMR": (800, 3000),
    "Metabolic Age": (18, 100),
    "Resting Heart Rate": (30, 220),
    "Body Cell Mass": (30, 150),
}

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Marks the "current" value on a card: Hume renders each card as
# "previous LABEL > current" (or some ordering of that), and Vision reads the
# trend arrow before the current value as a literal ">" or "›" character —
# either merged onto the value's own token or as a separate token in the same
# row. This is a more reliable signal for which value is current than
# position, since a card's label can render before or after its previous
# value depending on the report layout (see _resolve_previous_current()).
CURRENT_MARKER_RE = re.compile(r"[>›]")

# Minimum difflib.SequenceMatcher ratio for a fuzzy label match to be
# accepted (see _fuzzy_bind_metrics()). Calibrated against real OCR garbling
# ("Rodvca Macd" for "Body Cell Mass" scores 0.48) while staying clear of the
# next-closest wrong metric for the same garbled text (0.40) — the fuzzy pass
# also narrows the candidate pool to still-unbound metrics as it goes, which
# is what keeps a threshold this low from misfiring in practice.
FUZZY_LABEL_THRESHOLD = 0.45

# Extra comparison strings for metrics whose canonical label alone is a poor
# fuzzy target for how Hume actually renders (or OCR garbles) the card, e.g.
# "BMR" is too short to compare well against a garbled multi-word label.
FUZZY_ALIASES: dict[str, tuple[str, ...]] = {
    "BMR": ("bmr (basal metabolic rate)", "basal metabolic rate"),
}


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
    source_label: str | None = None
    """Raw OCR text the label was fuzzy-matched from; ``None`` for an exact match."""


@dataclass
class Report:
    """Parsed report plus the raw OCR lines used to build it."""

    metrics: list[Metric] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)


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


def _all_label_spans(
    row: list[Token], all_patterns: list[re.Pattern[str]]
) -> list[tuple[int, int]]:
    """Every metric label's token span that matches somewhere in a row.

    Used to find the boundaries between two metrics whose labels share a row
    (Hume's horizontal summary strip renders "Weight ... Metabolic Age ..."
    on one line), regardless of which metric is currently being bound.

    Args:
        row: A clustered OCR row, sorted left-to-right.
        all_patterns: Every metric's compiled label pattern.

    Returns:
        Matched (start_index, end_index) spans, in no particular order.
    """
    spans = []
    for pattern in all_patterns:
        span = _label_span(row, pattern)
        if span is not None:
            spans.append(span)
    return spans


def _numeric_candidates_by_boundary(
    row: list[Token],
    row_idx: int,
    own_span: tuple[int, int],
    all_spans: list[tuple[int, int]],
    used: set[tuple[int, int]],
) -> list[tuple[int, int, Token]]:
    """Unclaimed numeric tokens in a row whose nearest label is ``own_span``.

    A row can carry more than one metric's label (the summary strip). Each
    numeric token is assigned to whichever label span is closest to it by
    token-index distance — ties (a token equidistant between two labels) go
    to the preceding label, matching how Hume actually lays these rows out:
    "LABEL_A value LABEL_B value" puts each value right after its own label,
    not before the next one.

    Args:
        row: A clustered OCR row, sorted left-to-right.
        row_idx: This row's index, for building (row_idx, token_idx) keys.
        own_span: The token span of the label being bound.
        all_spans: Every label span found on this row (including ``own_span``).
        used: (row_index, token_index) pairs already claimed by another metric.

    Returns:
        ``(row_idx, token_idx, token)`` for each numeric token nearest to
        ``own_span``.
    """
    results: list[tuple[int, int, Token]] = []
    for i, token in enumerate(row):
        if (row_idx, i) in used:
            continue
        if any(s[0] <= i <= s[1] for s in all_spans):
            continue  # Part of a label, not a value.
        if not NUMBER_RE.search(token.text):
            continue
        nearest = min(
            all_spans,
            key=lambda s: (
                (s[0] - i) if s[0] > i else (i - s[1]),
                0 if s[1] < i else 1,  # Tie-break: prefer the preceding label.
            ),
        )
        if nearest == own_span:
            results.append((row_idx, i, token))
    return results


def _row_has_bare_marker(row: list[Token]) -> bool:
    """Whether a row carries a trend-arrow marker as its own standalone token."""
    return any(
        CURRENT_MARKER_RE.search(t.text) and not NUMBER_RE.search(t.text) for t in row
    )


def _resolve_previous_current(
    candidates: list[tuple[int, int, Token]], rows: list[list[Token]]
) -> tuple[str | None, str]:
    """Decide which candidate values are "previous" and "current".

    A candidate is "current" if its own token carries the trend-arrow marker
    (">value" merged by OCR into one token), or if it is the sole numeric
    candidate from a row that carries the marker as a separate token. This
    keys the decision off the marker Hume actually renders rather than
    position, since a card's previous value can render before or after its
    label depending on layout. Position is only used as a fallback when no
    candidate carries a marker (some cards render "previous LABEL current"
    with no visible arrow at all) or when the marker is ambiguous — the
    first candidate found is "previous", the last is "current".

    Args:
        candidates: Numeric candidates in scan order (own row first, then
            forward-window rows), as returned by
            :func:`_numeric_candidates_by_boundary` and its forward-window
            counterpart.
        rows: All clustered rows, for looking up a candidate's full row when
            checking for a standalone marker token.

    Returns:
        ``(previous, current)`` numeric strings; ``previous`` is ``None``
        when only one candidate was found.
    """
    if len(candidates) == 1:
        return None, NUMBER_RE.search(candidates[0][2].text).group()

    by_row: dict[int, list[int]] = {}
    for idx, (row_idx, _, _) in enumerate(candidates):
        by_row.setdefault(row_idx, []).append(idx)

    marked = [bool(CURRENT_MARKER_RE.search(c[2].text)) for c in candidates]
    for row_idx, idxs in by_row.items():
        solo_unmarked = len(idxs) == 1 and not marked[idxs[0]]
        if solo_unmarked and _row_has_bare_marker(rows[row_idx]):
            marked[idxs[0]] = True

    marked_candidates = [c for c, m in zip(candidates, marked, strict=True) if m]
    unmarked_candidates = [c for c, m in zip(candidates, marked, strict=True) if not m]

    if len(marked_candidates) == 1 and unmarked_candidates:
        previous_tok, current_tok = unmarked_candidates[0][2], marked_candidates[0][2]
    else:
        previous_tok, current_tok = candidates[0][2], candidates[-1][2]
    return (
        NUMBER_RE.search(previous_tok.text).group(),
        NUMBER_RE.search(current_tok.text).group(),
    )


def _gather_candidates(
    rows: list[list[Token]],
    label_idx: int,
    own_span: tuple[int, int],
    all_spans: list[tuple[int, int]],
    used: set[tuple[int, int]],
    pattern: re.Pattern[str],
    all_patterns: list[re.Pattern[str]],
) -> list[tuple[int, int, Token]]:
    """Gather a label occurrence's numeric candidates from its card window.

    Checks the label's own row first (boundary-aware, so a neighbouring
    label sharing the row cannot donate its values), then up to
    ``CARD_FORWARD_WINDOW`` rows below — mirroring the Hume card layout,
    where values render either beside the label or directly beneath it.
    Accumulation stops as soon as two candidates are found, or immediately
    if a forward row's text matches another metric's label (the window has
    run into the next card).

    Args:
        rows: Clustered OCR rows from :func:`cluster_rows`.
        label_idx: Row index of the label occurrence.
        own_span: The label's own token span on its row.
        all_spans: Every label span found on the label's own row.
        used: (row_index, token_index) pairs already claimed by another metric.
        pattern: This metric's compiled label pattern.
        all_patterns: Every metric's compiled label pattern.

    Returns:
        Numeric candidates in scan order, capped at two.
    """
    candidates: list[tuple[int, int, Token]] = []
    for offset in range(CARD_FORWARD_WINDOW + 1):
        row_idx = label_idx + offset
        if row_idx >= len(rows):
            continue
        row = rows[row_idx]
        if offset == 0:
            found = _numeric_candidates_by_boundary(
                row, row_idx, own_span, all_spans, used
            )
        else:
            text = " ".join(t.text for t in row).lower()
            if any(p is not pattern and p.search(text) for p in all_patterns):
                break  # Ran into the next card; stop accumulating.
            found = [
                (row_idx, i, t)
                for i, t in enumerate(row)
                if (row_idx, i) not in used and NUMBER_RE.search(t.text)
            ]
        candidates.extend(found)
        if len(candidates) >= 2:
            break
    return candidates[:2]


def _bind_metric(
    rows: list[list[Token]],
    pattern: re.Pattern[str],
    used: set[tuple[int, int]],
    all_patterns: list[re.Pattern[str]],
) -> tuple[str | None, str] | None:
    """Locate a metric's label and bind its previous/current values.

    Scans every unclaimed occurrence of ``pattern`` top to bottom. A row
    that carries more than one metric's label — Hume's horizontal summary
    strip, e.g. "Weight ... Metabolic Age ..." — is a compressed highlight,
    not the detailed card; it is kept as a fallback but a later occurrence
    on a row with only this metric's label is preferred whenever one binds
    successfully. A single-label occurrence that finds no value in its card
    window aborts the whole search immediately: the metric is abandoned
    rather than falling through to some unrelated row later in the document
    that happens to contain the same text (e.g. a footnote).

    Args:
        rows: Clustered OCR rows from :func:`cluster_rows`.
        pattern: The metric's compiled label pattern.
        used: (row_index, token_index) pairs already bound to another
            metric; mutated in place with the tokens this call claims.
        all_patterns: Every metric's compiled label pattern, used to detect
            a neighbouring label on the same row and a forward row that has
            crossed into the next card.

    Returns:
        ``(previous, current)``, or ``None`` if the label was not found or no
        value bound to it.
    """
    fallback: tuple[int, tuple[int, int], list[tuple[int, int, Token]]] | None = None

    for label_idx, label_row in enumerate(rows):
        span = _label_span(label_row, pattern)
        if span is None:
            continue
        start_idx, end_idx = span
        if any((label_idx, i) in used for i in range(start_idx, end_idx + 1)):
            continue  # This occurrence's tokens are already another metric's.

        all_spans = _all_label_spans(label_row, all_patterns)
        is_shared_row = len(all_spans) > 1
        candidates = _gather_candidates(
            rows, label_idx, span, all_spans, used, pattern, all_patterns
        )

        if not is_shared_row:
            if not candidates:
                return None  # Label found but no value in window: abandon it.
            return _finalize_binding(rows, label_idx, span, candidates, used)

        if candidates and fallback is None:
            fallback = (label_idx, span, candidates)

    if fallback is not None:
        return _finalize_binding(rows, *fallback, used)
    return None


def _finalize_binding(
    rows: list[list[Token]],
    label_idx: int,
    span: tuple[int, int],
    candidates: list[tuple[int, int, Token]],
    used: set[tuple[int, int]],
) -> tuple[str | None, str]:
    """Claim a binding's tokens and resolve its previous/current values."""
    start_idx, end_idx = span
    for i in range(start_idx, end_idx + 1):
        used.add((label_idx, i))
    for row_idx, token_idx, _ in candidates:
        used.add((row_idx, token_idx))
    return _resolve_previous_current(candidates, rows)


def _fuzzy_bind_metrics(
    rows: list[list[Token]],
    used: set[tuple[int, int]],
    already_bound: set[str],
) -> list[Metric]:
    """Bind still-unmatched metrics by fuzzy-comparing garbled labels.

    Exact substring/regex matching cannot survive OCR garbling as severe as
    "Boov -at Masd" for "Body Fat Mass". This pass compares every unclaimed
    non-numeric token against each not-yet-bound metric's canonical label
    (:data:`FUZZY_ALIASES` gives extra comparison strings where the label
    text is too dissimilar to compare well, e.g. "BMR"), and accepts the
    best match if it clears :data:`FUZZY_LABEL_THRESHOLD`. Candidate metrics
    narrow to those still unbound as the scan proceeds, which is what keeps
    a low threshold from misfiring — an already-bound metric's canonical
    label can no longer steal a garbled token meant for another metric.

    Only same-row values are considered (no forward-window search): a label
    too garbled to match a known phrase is already a low-confidence bind,
    and Hume's card layout renders the label's own value(s) on its own row
    in every observed case.

    Args:
        rows: Clustered OCR rows from :func:`cluster_rows`.
        used: (row_index, token_index) pairs already bound; mutated in place.
        already_bound: Labels already bound by exact matching.

    Returns:
        Newly bound metrics, each carrying the raw OCR text it matched in
        ``Metric.source_label`` so a bad match is visible in the output.
    """
    remaining = {
        label: unit for label, _, unit in METRIC_SPECS if label not in already_bound
    }
    all_patterns = [pattern for _, pattern, _ in METRIC_SPECS]
    matched: list[Metric] = []

    for row_idx, row in enumerate(rows):
        for tok_idx, token in enumerate(row):
            if not remaining:
                return matched
            if (row_idx, tok_idx) in used or NUMBER_RE.search(token.text):
                continue
            if " " not in token.text.strip():
                continue  # A lone word is too short to fuzzy-match reliably.
            if any(p.search(token.text.lower()) for p in all_patterns):
                continue  # A genuine (if unclaimed leftover) label, not garbled.

            best_label, best_score = None, 0.0
            for label in remaining:
                aliases = FUZZY_ALIASES.get(label, (label.lower(),))
                score = max(
                    difflib.SequenceMatcher(None, token.text.lower(), alias).ratio()
                    for alias in aliases
                )
                if score > best_score:
                    best_label, best_score = label, score
            if best_label is None or best_score < FUZZY_LABEL_THRESHOLD:
                continue

            own_span = (tok_idx, tok_idx)
            candidates = _numeric_candidates_by_boundary(
                row, row_idx, own_span, [own_span], used
            )
            if not candidates:
                continue

            used.add((row_idx, tok_idx))
            for r_idx, t_idx, _ in candidates:
                used.add((r_idx, t_idx))
            previous, current = _resolve_previous_current(candidates, rows)
            unit = remaining.pop(best_label)
            matched.append(
                Metric(best_label, previous, current, unit, source_label=token.text)
            )
    return matched


def parse_metrics(rows: list[list[Token]]) -> list[Metric]:
    """Map known Hume labels to their previous/current values.

    Runs exact word-boundary matching first (:func:`_bind_metric`), then a
    fuzzy pass (:func:`_fuzzy_bind_metrics`) over whatever remains
    unmatched, to recover labels OCR has garbled beyond exact matching.

    Args:
        rows: Clustered OCR rows from :func:`cluster_rows`.

    Returns:
        Parsed metrics in canonical order, skipping any not found or whose
        values could not be bound within the card window.
    """
    metrics: dict[str, Metric] = {}
    used: set[tuple[int, int]] = set()
    all_patterns = [pattern for _, pattern, _ in METRIC_SPECS]
    for label, pattern, unit in METRIC_SPECS:
        result = _bind_metric(rows, pattern, used, all_patterns)
        if result is None:
            continue
        previous, current = result
        metrics[label] = Metric(label, previous, current, unit)

    for metric in _fuzzy_bind_metrics(rows, used, set(metrics)):
        metrics[metric.label] = metric

    return [metrics[label] for label, _, _ in METRIC_SPECS if label in metrics]


def _repair_unitless_value(raw: str, bounds: tuple[float, float]) -> str | None:
    """Recover a trend-arrow-corrupted value for a unit-less metric.

    Every Hume card carries a trend arrow (⇆ ↑ ↓) after its current value.
    For metrics with a unit, the unit string delimits the value from that
    arrow so Vision keeps them as separate tokens. Unit-less metrics (e.g.
    Visceral Fat Index) have no such delimiter, so Vision sometimes merges
    the arrow glyph straight onto the value's last digit (``7`` + misread
    arrow becomes the single token ``75``). Stripping that trailing
    character recovers the original value when doing so lands back inside
    the metric's plausible range.

    This is an interim, single-character fix — see issue #5's proposed
    x-geometry filter (discarding tokens whose x-position falls in the
    arrow column) for the general fix, which should replace this once
    landed.

    Args:
        raw: The corrupted numeric string as extracted from OCR.
        bounds: The metric's plausibility range.

    Returns:
        The repaired numeric string, or ``None`` if stripping the last
        character does not produce a plausible number.
    """
    stripped = raw[:-1]
    if not NUMBER_RE.fullmatch(stripped):
        return None
    low, high = bounds
    if not (low <= float(stripped) <= high):
        return None
    return stripped


def validate_metrics(metrics: list[Metric]) -> list[str]:
    """Flag implausible metrics in place and return human-readable warnings.

    Applies per-metric plausibility ranges (:data:`METRIC_RANGES`) and
    cross-metric invariants that catch card-crossover errors a range check
    alone would miss (e.g. a metabolic age bound to a weight reading). Any
    metric that fails a check has ``Metric.unverified`` set so the renderer
    can flag it instead of reporting a wrong number as good data. Unit-less
    metrics get one more chance first: :func:`_repair_unitless_value` tries
    to recover a value corrupted by a merged trend-arrow glyph before giving
    up and flagging it.

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
            if m.unit is None:
                repaired = _repair_unitless_value(m.current, bounds)
                if repaired is not None:
                    m.current = repaired
                    continue
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

    skeletal_muscle, bone = value("Skeletal Muscle Mass"), value("Bone Mass")
    if skeletal_muscle is not None and bone is not None and not skeletal_muscle > bone:
        flag_pair(
            "Skeletal Muscle Mass",
            "Bone Mass",
            f"Skeletal Muscle Mass ({skeletal_muscle}) should be greater than "
            f"Bone Mass ({bone})",
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

    fuzzy = [m for m in report.metrics if m.source_label]
    if not raw_only and fuzzy:
        lines += ["## Fuzzy-matched labels", ""]
        lines += [f'- {m.label} ← "{m.source_label}"' for m in fuzzy]
        lines += [
            "",
            "> These labels didn't match a known Hume phrasing exactly and were "
            "bound by approximate text similarity — verify against the source "
            "image before trusting the value.",
            "",
        ]

    if not raw_only and report.unparsed:
        lines += ["## Metrics not parsed", ""]
        lines += [f"- {label}" for label in report.unparsed]
        lines += [
            "",
            "> Expected but not bound to a value in this report — see the raw "
            "OCR text below to check whether the source image renders them "
            "differently than usual.",
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
    parsed_labels = {m.label for m in metrics}
    unparsed = [label for label, _, _ in METRIC_SPECS if label not in parsed_labels]
    report = Report(
        metrics=metrics,
        raw_lines=[" ".join(t.text for t in row) for row in rows],
        unparsed=[] if args.raw else unparsed,
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
