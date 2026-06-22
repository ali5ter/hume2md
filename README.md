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
```

## Output

A Markdown table of recognized metrics (Weight, Body Fat %, Skeletal Muscle
Mass, Body Water %, BMR, Metabolic Age, …) followed by a `Raw OCR text`
section.

## Limitations

- The previous/current column order is inferred from horizontal position
  (left = previous, right = current). **Verify against the source image** —
  Hume's layout can shift between report types.
- Parsing is best-effort; the raw OCR section is always included so the export
  is useful even when a field is missed. Use `--raw` if parsing misbehaves.
- BIA body-water and muscle figures are noisy between scans. Treat single
  readings with caution and watch multi-week trends (Hume's own guidance).

## License

MIT — see [LICENSE](LICENSE).
