# Reproducible runbook

## Environment

- Python 3.11 or later
- `pip install -r requirements.txt`
- NumPy 2.2.6, pandas 2.3.3, Matplotlib 3.11.0 for the verified local run

## Command

Run from this directory:

```powershell
Get-ChildItem code -Filter *.py | Sort-Object Name | ForEach-Object { python $_.FullName }
```

`orientation_stochastic_contact_model.py` reads the effective-contact proxy
output, so the alphabetic order above is recommended. The conceptual figure
script writes to the submission-package figure folder only when used from that
package; it is retained here as transparent source code.

## Acceptance checks

- `costa2024_benchmark_predictions.csv`: maximum absolute angle error < 2.5°.
- `costa2024_identifiability_pairs.csv`: equal geometry ratio and different
  observed sliding fractions.
- `orientation_stochastic_contact_predictions.csv`: held-out error < 0.05 and
  below the pooled scalar negative-control error.
- `orientation_stochastic_contact_sensitivity.csv`: proxy-spread sensitivity
  remains visible rather than hidden.

## Provenance

`DATASETS_AND_LINKS.csv` lists article DOI or official URLs. The publisher
supplement is not redistributed. `REPRO_MANIFEST.csv` hashes every included
source, input, and derived output file.
