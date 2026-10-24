# Dirty-feature-value experiments and reproducible analyses

Version 1.2.1 adds the analysis code and fixed design specifications for
representation, placement, fingerprint, outlier-component and categorical-interface controls.
The optional plotting entry point includes all required helpers and archived
numeric estimates; it does not require manuscript files or historical figure assets.

## Reproduce the archived results

Download and extract the matching result-data archive from
https://doi.org/10.5281/zenodo.21979275, then run with Python 3.11:

```sh
python -m pip install -r extensions/requirements.txt
python extensions/reproduce.py --data-root /path/to/tfm-dirty-values-results
```

The command verifies data hashes and reconstructs earlier extensions from scalar
records and component-control metrics from saved probabilities. It compares all
recomputed summary tables with the archived references. Outputs are written to
dedicated `reproduced` folders in the extracted data archive. It does not refit models.

For plots, install `extensions/requirements-figures.txt` and append `--figures`.
If the analyses have already been reconstructed, use `--figures-only` instead.
Use `--verify-only` to check the data files without analysis.

## Refit the component controls

The optional `extensions/refit/run.py` fits the frozen component jobs.
It requires the original split arrays and pinned models identified in
`data/DATASETS.md`, `MODEL_ARTIFACTS.md` and `pyproject.toml`.
Install the model environment with `uv sync --frozen`, activate that
environment, and run each shard in turn:

```sh
python extensions/refit/run.py --splits-root /path/to/splits --model-root /path/to/models --output-root /path/to/new-run --shard cpu
python extensions/refit/run.py --splits-root /path/to/splits --model-root /path/to/models --output-root /path/to/new-run --shard gpu
python extensions/analyze_components.py --root /path/to/new-run
```

Split and checkpoint hashes must match. Use a separate output directory.
Model refits are not guaranteed to be bitwise identical across executions;
the archived-probability reconstruction above is the exact reporting boundary.
Historical orchestration and original database analyses remain in tag v1.1.0
and https://doi.org/10.5281/zenodo.22099547.

## Data and licenses

The companion archive contains scalar records and anonymous binary labels,
predictions and masks for the new component controls. Raw feature tables and
third-party checkpoints are not distributed. Source code: MIT; original result
data: CC BY 4.0. Citation metadata: `CITATION.cff`.

Software archive: https://doi.org/10.5281/zenodo.21979277.
