# hume2md

Convert a Hume Body Pod progress-report PNG into labeled Markdown.

Hume Health exports its body-composition progress reports only as PNG
screenshots. Those are awkward to read programmatically and easy to misread by
eye. `hume2md` runs on-device [Apple Vision](https://developer.apple.com/documentation/vision)
OCR over the image, parses each metric's previous/current values, and writes a
Markdown summary — with the raw OCR text always appended as a fallback.

## Requirements

- macOS (the OCR uses the Apple Vision framework via `ocrmac`)
- Python 3.10+

## Installation

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ocrmac rich
```

Optionally make the script directly executable:

```bash
chmod +x hume2md.py
```

## Usage

If you ran `chmod +x`:

```bash
./hume2md.py HumeProgressReport22_Jun_2026.png
```

Otherwise call it through the interpreter (no execute bit needed):

```bash
python3 hume2md.py HumeProgressReport22_Jun_2026.png
```

Writes `HumeProgressReport22_Jun_2026.md` next to the image. Options:

```bash
python3 hume2md.py REPORT.png -o out.md      # choose output path
python3 hume2md.py REPORT.png --raw          # raw OCR text only (skip parsing)
python3 hume2md.py REPORT.png --date 2026-06-22
python3 hume2md.py REPORT.png --csv metrics.csv  # append-ready CSV rows
```

## Output

A Markdown table of recognized metrics (Weight, Body Fat %, Skeletal Muscle
Mass, Body Water %, BMR, Metabolic Age, …) followed by a `Raw OCR text`
section. With `--csv`, each verified metric is also written as a
`date,metric,value,unit` row.

Each metric is bound to its own card: the label is located, then its
previous/current values are read from the numeric tokens nearest that label's
own position — not from list position — so a neighboring card's numbers
cannot bleed in. Parsed values are then checked against per-metric
plausibility ranges and cross-metric invariants (e.g. body cell mass < lean
mass < weight). A metric that fails a check is rendered as `⚠️ unverified`
in the Markdown, a warning is printed to stderr, and the process exits
non-zero — a suspect value is never silently reported as good data, and is
excluded from `--csv` output.

## Limitations

- Parsing is best-effort; the raw OCR section is always included so the export
  is useful even when a field is missed. Use `--raw` if parsing misbehaves.
- Plausibility ranges and invariants catch out-of-range or physically
  impossible values, not every misread — a wrong-but-plausible number can
  still slip through. **Verify against the source image.**
- BIA body-water and muscle figures are noisy between scans. Treat single
  readings with caution and watch multi-week trends (Hume's own guidance).

## Exit codes

| Code | Meaning |
| ---- | ------- |
| 0 | Success |
| 1 | Input/usage error |
| 2 | OCR/runtime error |
| 3 | One or more metrics failed validation (Markdown/CSV are still written) |

## Testing

```bash
pip install pytest
pytest
```

Regression tests run against committed OCR-token JSON fixtures under
`tests/fixtures/`, so they don't require macOS or Apple Vision.

## License

MIT — see [LICENSE](LICENSE).
