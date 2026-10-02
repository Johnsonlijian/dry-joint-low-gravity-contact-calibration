# Dry-joint contact-state variability under reduced gravity

This repository contains the reproducible analysis assets for the manuscript
**Contact-state variability and stack-height dilution govern when reduced gravity
changes dry-joint stability**.

It contains source scripts, clearly labelled input tables, derived
CSV/PNG/SVG outputs, a pinned Python environment, and source links. It
intentionally excludes the manuscript, cover letter, submission records,
internal audits, private author information, and the publisher-supplied Costa
workbook because redistribution rights and the publisher's supplementary-file
licence do not permit us to redistribute that workbook here.

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
`REPRO_MANIFEST.csv` for provenance and file fingerprints. The repository is
the code/data companion for the manuscript; it is not itself evidence of
external validation or journal acceptance.

## Release boundary

The repository is released under the MIT licence for the code and derived
non-sensitive outputs. Third-party source data remain linked, not
redistributed. The canonical public remote is
`https://github.com/Johnsonlijian/dry-joint-low-gravity-contact-calibration`.
An archival Zenodo record will be linked here once its public DOI is minted.
