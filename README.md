# Dirty Feature Values in Tabular Foundation Models

Reproducibility release **v1.3.0** (2026-10-24).

Versioned software: https://doi.org/10.5281/zenodo.23620181  
Matching result data: https://doi.org/10.5281/zenodo.23620184

This release adds a fixed descriptive CatBoost comparison to the existing
pipeline audit. It contains code and configurations; use the matching result
data archive for scalar records and prediction vectors. The original H1–H3
test families are unchanged. MIT applies to software; CC BY 4.0 to original
result data. Raw feature tables and third-party checkpoints are not included.

## Replay the CatBoost correction (no fitting or network)

Use **Python 3.11.9** and the dedicated metric dependencies:

```sh
python -m pip install -r extensions/catboost_correction/requirements.txt
python extensions/catboost_correction/replay_correction.py --root /path/to/tfm-dirty-values-results/catboost_correction_bundle
```

The bundle contains `correction/`; the historical injector ZIP is retained
unchanged in `archives/`. All 915 record hashes and 595 retained source-byte
identities are checked. Four metrics for 320 corrective prediction vectors
must replay exactly. The other 595 records contain scalar outcomes rather
than prediction vectors; their verification does not constitute refitting.

## Reproduce dataset-first CatBoost tables

```sh
python extensions/catboost_correction/analyze_catboost.py --data-root /path/to/tfm-dirty-values-results --output /path/to/new-analysis
```

The output directory must be new. This rebuilds the six checked CSV tables
using all five original pipelines and CatBoost, five seeds averaged within
dataset, and fixed unadjusted dataset-bootstrap intervals. To redraw the two
CatBoost figures, install `requirements-figures.txt` from the same directory
and append `--figures`. No original hypothesis tests are reopened.

## Preceding studies

`README-v1.2.1.md` documents the retained reconstruction/refit entry points.
Run `python extensions/reproduce.py --data-root /path/to/tfm-dirty-values-results`
for earlier scalar and component analyses. Refit entry points are explicit
actions requiring separately obtained input arrays and model checkpoints.
The core model dependency lock and package version are retained; the v1.3.0
archive tag is distinct from the internal Python package version.

Software concept DOI: https://doi.org/10.5281/zenodo.21979277  
Data concept DOI: https://doi.org/10.5281/zenodo.21979275

Previous exact software DOI: https://doi.org/10.5281/zenodo.23384468  
Previous exact data DOI: https://doi.org/10.5281/zenodo.23383120

