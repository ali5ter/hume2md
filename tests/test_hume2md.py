"""Regression tests for hume2md's card-based metric parser.

Fixtures are OCR-token JSON files (see tests/fixtures/), not PNGs, so the
suite runs without macOS or Apple Vision. Each covers a failure mode from the
parser rewrite: cross-card contamination, an abandoned metric that must not
fall back to an unrelated row, implausible values caught by validation, and
(regenerated_report) the layouts from issue #3 — a label flanked by
previous/current on one row, and previous/current split across two rows by
OCR y-jitter.

Author: Alister Lewis-Bowen <alister@lewis-bowen.org>
"""

from __future__ import annotations

from pathlib import Path

from conftest import load_tokens

from hume2md import (
    METRIC_SPECS,
    Metric,
    cluster_rows,
    parse_metrics,
    render_markdown,
    validate_metrics,
    write_csv,
)
from hume2md import Report as ReportType


def _parse(fixture_name: str) -> list[Metric]:
    rows = cluster_rows(load_tokens(fixture_name))
    return parse_metrics(rows)


def test_typical_report_binds_every_metric_to_its_own_card():
    metrics = {m.label: m for m in _parse("typical_report")}

    assert set(metrics) == {
        "Health Score",
        "Body Fat Mass",
        "Subcutaneous Fat Mass",
        "Lean Mass",
        "Skeletal Muscle Mass",
        "Bone Mass",
        "Visceral Fat Index",
        "Body Water %",
        "Protein",
        "BMR",
        "Metabolic Age",
        "Resting Heart Rate",
        "Weight",
        "Body Cell Mass",
    }


def test_body_fat_mass_not_bound_to_subcutaneous_card():
    metrics = {m.label: m for m in _parse("typical_report")}

    assert metrics["Body Fat Mass"].current == "21.4"
    assert metrics["Subcutaneous Fat Mass"].current == "18.4"


def test_shared_row_labels_bind_independently_not_swapped():
    metrics = {m.label: m for m in _parse("typical_report")}

    assert metrics["Weight"].current == "176.7"
    assert metrics["Weight"].previous is None
    assert metrics["Metabolic Age"].current == "45"
    assert metrics["Metabolic Age"].previous is None


def test_bone_mass_not_confused_with_skeletal_muscle_mass():
    metrics = {m.label: m for m in _parse("typical_report")}

    assert metrics["Bone Mass"].current == "6.2"
    assert metrics["Skeletal Muscle Mass"].current == "98.7"


def test_typical_report_passes_validation():
    metrics = _parse("typical_report")

    assert validate_metrics(metrics) == []
    assert all(not m.unverified for m in metrics)


def test_label_with_no_numbers_in_window_is_abandoned_not_matched_later():
    metrics = _parse("abandoned_metric")

    assert metrics == []


def test_validation_flags_out_of_range_metabolic_age():
    metrics = {m.label: m for m in _parse("validation_failures")}
    warnings = validate_metrics(list(metrics.values()))

    assert metrics["Metabolic Age"].unverified is True
    assert any("Metabolic Age" in w for w in warnings)


def test_validation_flags_body_cell_mass_exceeding_lean_mass():
    metrics = {m.label: m for m in _parse("validation_failures")}
    warnings = validate_metrics(list(metrics.values()))

    assert metrics["Body Cell Mass"].unverified is True
    assert metrics["Lean Mass"].unverified is True
    assert any("Body Cell Mass" in w for w in warnings)


def test_validation_leaves_plausible_metrics_unflagged():
    metrics = {m.label: m for m in _parse("validation_failures")}
    validate_metrics(list(metrics.values()))

    assert metrics["Weight"].unverified is False


def test_render_markdown_shows_unverified_warning_for_flagged_metric():
    metrics = _parse("validation_failures")
    validate_metrics(metrics)
    report = ReportType(metrics=metrics, raw_lines=["dummy"])

    markdown = render_markdown(report, "report.png", "2026-08-23", raw_only=False)

    assert "⚠️ unverified" in markdown
    assert "| Weight | " in markdown
    assert "176.0" in markdown


def test_same_row_card_binds_own_values_not_the_next_cards():
    """Regression for issue #3: a label flanked by prev/current on one row
    must not fall through to the card below it (Skeletal Muscle Mass reading
    Bone Mass's value)."""
    metrics = {m.label: m for m in _parse("regenerated_report")}

    assert metrics["Skeletal Muscle Mass"].previous == "98.7"
    assert metrics["Skeletal Muscle Mass"].current == "98.8"
    assert metrics["Bone Mass"].previous == "23.0"
    assert metrics["Bone Mass"].current == "23.1"


def test_previous_and_current_split_across_rows_both_bind():
    """Regression for issue #3: OCR row-clustering can split "previous" and
    "current" a fraction of a pixel apart in y, landing them in separate
    rows. Both values must still be found, in order, not just the first."""
    metrics = {m.label: m for m in _parse("regenerated_report")}

    assert metrics["Weight"].previous == "175.9"
    assert metrics["Weight"].current == "176.7"
    assert metrics["Lean Mass"].previous == "143.6"
    assert metrics["Lean Mass"].current == "143.7"


def test_regenerated_report_passes_validation():
    metrics = _parse("regenerated_report")

    assert validate_metrics(metrics) == []
    assert all(not m.unverified for m in metrics)


def test_unparsed_metrics_are_listed_not_silently_dropped():
    rows = cluster_rows(load_tokens("regenerated_report"))
    metrics = parse_metrics(rows)
    parsed = {m.label for m in metrics}
    unparsed = [label for label, _, _ in METRIC_SPECS if label not in parsed]
    report = ReportType(metrics=metrics, raw_lines=["dummy"], unparsed=unparsed)

    markdown = render_markdown(report, "report.png", "2026-08-23", raw_only=False)

    assert "## Metrics not parsed" in markdown
    assert "- Body Fat Mass" in markdown
    assert "Skeletal Muscle Mass" not in markdown.split("## Metrics not parsed")[1]


def test_write_csv_omits_unverified_rows(tmp_path: Path):
    metrics = _parse("validation_failures")
    validate_metrics(metrics)
    out = tmp_path / "metrics.csv"

    written = write_csv(metrics, "2026-08-23", out)

    content = out.read_text()
    assert written == 1
    assert "Weight" in content
    assert "Metabolic Age" not in content
    assert "Body Cell Mass" not in content
    assert "Lean Mass" not in content
