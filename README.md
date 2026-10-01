# Public reproducibility repository draft

This repository contains the reproducible analysis assets for the manuscript
**Contact-state variability and stack-height dilution govern when reduced gravity
changes dry-joint stability**.

It is a local preparation tree. It contains source scripts, clearly labelled
input tables, derived CSV/PNG/SVG outputs, a pinned Python environment, and
source links. It intentionally excludes the manuscript, cover letter,
submission records, internal audits, private author information, and the
publisher-supplied Costa workbook because redistribution rights and author
approval are not yet confirmed.

## Reproduce

```powershell
python -m pip install -r requirements.txt
Get-ChildItem code -Filter *.py | Sort-Object Name | ForEach-Object { python $_.FullName }
```

The expected checks are a 2.34 degree maximum dominant-angle error and a held
out perpendicular N=4 sliding-fraction absolute error of 0.008871 for the
orientation-specific diagnostic versus 0.095328 for the pooled scalar negative
control. These are internal benchmark diagnostics, not external validation or
lunar joint measurements.

See `DATASETS_AND_LINKS.csv`, `REPRODUCIBLE_RUNBOOK.md`, and
`REPRO_MANIFEST.csv` for provenance and file fingerprints.

## Release boundary

The repository is `LOCAL_ONLY / NOT_PUBLISHED`. Before any public release,
verify the author list, license compatibility, third-party data rights, DOI
metadata, and the final manuscript version. The intended GitHub remote is
`https://github.com/Johnsonlijian/dry-joint-low-gravity-contact-calibration.git`;
no remote repository has been created or pushed by this task.
