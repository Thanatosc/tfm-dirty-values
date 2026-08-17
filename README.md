# Beyond Missingness: Dirty Feature Value Audit

This repository contains the experiment and analysis software for the study
*Beyond Missingness: A Preregistered Audit of Tabular Foundation Models Under
Dirty Feature Values*.

The frozen study evaluated TabPFN v2, TabICL v2, XGBoost, LightGBM, and Random
Forest on 11 OpenML binary-classification records. Five wrong-but-present feature
corruptions were applied at four affected-cell rates. The confirmatory analysis
treated datasets, not seeds or individual tasks, as the independent units.

At the primary 0.50 rate, both tabular foundation models degraded under all five
corruptions, and all five degradation slopes were positive under the frozen
tests. No TFM-versus-tree family contrast crossed the preregistered alpha
threshold. That non-detection is not evidence of equivalence.

## Repository scope

This software release includes:

- corruption operators, model adapters, metrics, persistent workers, and analysis code;
- stop, resume, status, and deterministic result-verification tools;
- the frozen experiment configurations in portable, path-normalized form;
- Phase 1 and Phase 2 preregistration protocols and the registered ECE key amendment;
- dataset, split, task, checkpoint, and runtime identity manifests;
- the frozen aggregate analysis and generated tables and figures; and
- unit and integration tests.

The companion Zenodo data record contains privacy-sanitized SQLite copies of the
126-task Kill Test and the complete 4,575-task Phase 2 run. Raw OpenML downloads,
processed feature matrices, model weights, runtime logs, manuscript text, review
files, and internal research-management records are not distributed. See
`RELEASE_SCOPE.md` for the exact boundary.

- Software version DOI: https://doi.org/10.5281/zenodo.21979278
- Software concept DOI: https://doi.org/10.5281/zenodo.21979277
- Result-data version DOI: https://doi.org/10.5281/zenodo.21979276
- Result-data concept DOI: https://doi.org/10.5281/zenodo.21979275

## Environment

Python 3.11 and the complete dependency graph are pinned in `uv.lock`. The two
TFM adapters require a CUDA-capable environment for the formal configuration.
Install with [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --extra dev
```

Model checkpoints are not redistributed. Download and verify the files listed in
`MODEL_ARTIFACTS.md` before running model experiments.

## Tests

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_tests.ps1
```

The release baseline is 45 passing tests.

## Verify the frozen aggregate package

The generated tables and figures are hash-bound to the frozen analysis:

```powershell
uv run python scripts\generate_paper_results.py --check
```

To recompute the aggregate analysis from the companion public SQLite database,
place `state-public.sqlite3` at `results/full_study/state.sqlite3` and run:

```powershell
uv run python scripts\reanalyze_archived_results.py
```

The command refuses output unless all 4,575 task identities are present and the
recomputed JSON has SHA-256
`7D9389384762BC04BD352F0540F225D05D92B8F60B99D309232ADC0D9EBBFE30`.

## Experiment controls

The workers persist task state in SQLite, write each task output atomically, and
support checkpointed stop and resume:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\full_study_status.ps1 -Json
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stop_full_study.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\resume_full_study.ps1 -ConfirmResume
```

Starting a new outcome-producing run requires a separate local authorization
artifact. The formal study's private authorization files are intentionally not
part of this release.

## Citation and licenses

Citation metadata is provided in `CITATION.cff`. Source code is licensed under
MIT. Original aggregate result files under `artifacts/`, `paper/results/`, and
`results/`, together with the companion result-data record, are licensed under
CC BY 4.0; see `LICENSE-DATA`.

Author: Siyu Cai, School of Computing and Artificial Intelligence, Southwest
Jiaotong University. ORCID: 0009-0003-3716-0008.
