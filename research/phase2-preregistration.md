# Phase 2 Full-Study Preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-14
- Verification Status: UNVERIFIED
- Version Label: phase2_prereg_v1
- Outcome boundary: no Phase 2 model outcome was accessed when this document was frozen

## Objective

This confirmatory benchmark audits how two pinned tabular foundation models (TFMs) and three tree baselines degrade when observed feature values are corrupted. It extends the Phase 1 Kill Test to a frozen 11-record OpenML panel and five corruption operators. The study is an evaluation of frozen models; it does not train a new model, propose a repair algorithm, or make deployment-level causal claims.

## Research questions and estimands

- **RQ1:** How does ROC-AUC degradation vary across measurement noise, outliers, quantization, unit errors, and categorical encoding errors as the affected-cell rate increases?
- **RQ2:** Is the average degradation of the two TFMs different from the average degradation of the three tree baselines?
- **RQ3:** Is the degradation-rate curve increasing over the four affected-cell rates?
- **RQ4:** Do balanced accuracy, log loss, and ECE show the same direction as ROC-AUC, and does calibration deteriorate before discrimination?

The independent unit is the dataset. For each dataset-condition-model, the five seed results are averaged arithmetically before any dataset-level test. Seeds and corruption instances are not treated as independent observations.

## Frozen panel

- OpenML IDs: `3, 24, 31, 44, 802, 1049, 1487, 1504, 42193, 44158, 44160`.
- The panel is deduplicated by the frozen underlying record identity in `data/full-panel.json`.
- Selection constraints: 1,000-10,000 rows, 10-100 predictors, binary target, native missing fraction below 0.05, and a reusable public license.
- C5 categorical panel: `3, 24, 31, 802, 42193, 44158, 44160` (at least three categorical predictors each).
- Numeric panel: the other nine records. All numeric operators are restricted to this panel; the two all-categorical datasets receive clean baselines and C5 tasks only.
- COMPAS (`42193`) is used only as a predictive robustness benchmark. No fairness, policy, or individual-level conclusion is permitted.
- Eight previously missing citation fields were refreshed from OpenML lineage and verified scholarly metadata before this preregistration.

## Models and immutable versions

- TabPFN v2: `tabpfn==2.2.1`, frozen checkpoint hash recorded in `configs/full_study.yaml`.
- TabICLv2: `tabicl==2.1.1`, frozen checkpoint hash recorded in `configs/full_study.yaml`.
- XGBoost: `xgboost==3.2.0`.
- LightGBM: `lightgbm==4.7.0`.
- Random Forest: `scikit-learn==1.6.1`, 500 trees.
- TFMs use their frozen checkpoints and preregistered estimator counts. Tree models use the frozen settings in the configuration and no post hoc tuning.

## Splits and seeds

Each dataset uses a stratified 70/30 context-query split for seeds `[1907, 2027, 4049, 6029, 8089]`. Every clean and polluted condition for a dataset-seed pair reuses the same split. Category mappings and corruption parameters are fitted on the clean context only. Context and query corruption use distinct deterministic RNG streams. Labels never enter corruption selection.

## Corruption protocol

The affected-cell rates are `[0.05, 0.15, 0.30, 0.50]`. The rate is the proportion of observed eligible cells, not a claim about an operator's physical magnitude. Operator magnitudes are fixed before Phase 2 execution at the values carried forward from the frozen Kill Test:

| Operator | Fixed parameters | Eligible cells |
|---|---|---|
| Measurement noise | Gaussian noise, `sigma_ratio=0.50` | observed numeric cells |
| Outlier | mean plus or minus `10 * context SD` | observed numeric cells |
| Quantization | equal-width, 8 bins | observed numeric cells |
| Unit error | random factor from `{1000, 0.001}` | observed numeric cells |
| Categorical error | mixed swap/novel, `novel_level_fraction=0.50` | observed categorical cells |

The clean baseline is one task per dataset-seed-model. Corruption parameters are not re-estimated from the query split.

## Frozen task count

- Clean: `11 datasets * 5 seeds * 5 models = 275`.
- Numeric operators: `9 datasets * 4 operators * 4 rates * 5 seeds * 5 models = 3,600`.
- Categorical operator: `7 datasets * 4 rates * 5 seeds * 5 models = 700`.
- Total: **4,575 unique tasks**. The task identity and complete task list are frozen in `artifacts/phase2/full-study-task-manifest.json` before authorization.

## Outcomes

Primary metric: ROC-AUC degradation, defined as `clean ROC-AUC - polluted ROC-AUC`. Secondary metrics are balanced-accuracy degradation, log-loss increase, and ECE increase. Model provenance, split hashes, corruption masks, and input hashes are persisted with every output.

## Confirmatory analysis

At the primary rate of 0.50, the following tests are confirmatory:

1. **H1:** the mean degradation of the two TFMs differs from zero, tested with a two-sided exact sign-flip test on dataset aggregates.
2. **H2:** the mean TFM degradation minus mean tree-baseline degradation differs from zero, using the same paired dataset-level sign-flip test. Negative values mean the TFMs degrade less.
3. **H3:** the mean slope of ROC-AUC degradation over the four rates, averaged across the five models within each dataset, differs from zero.

Holm correction is applied within each RQ and eligibility panel: the four numeric corruptions form one family and categorical error is a separate singleton family. Individual model contrasts are exploratory effect sizes with 95% dataset bootstrap intervals, not additional confirmatory claims. Bootstrap resampling is by dataset, with 10,000 resamples and seed `260814`.

RQ4 and the ECE-before-AUC question are secondary descriptive analyses. A safe-rate statement is reported only when the preregistered effect size and bootstrap interval support it; no threshold is selected after seeing results.

## Missing, failed, and anomalous tasks

All 4,575 frozen tasks are required for confirmatory analysis. A task that fails after the permitted deterministic retry remains a disclosed failure and blocks confirmatory analysis; no failed output is substituted or silently deleted. A worker reclaim or interruption is recorded in SQLite and event logs, and only the interrupted task may be requeued after the worker lock is reacquired.

## Checkpoint, resume, and monitoring protocol

- The persistent queue is SQLite-backed and protected by a single-worker process lock. A task output is atomically written before its SQLite row is marked `completed`.
- The heartbeat file is refreshed every 15 seconds and records worker PID, current task, current-task start time, heartbeat age, and RSS memory. A heartbeat older than 120 seconds, a missing worker process, or RSS above three times the launch value is an advisory alert.
- `scripts/full_study_status.ps1 -Watch -IntervalSeconds 30` is the canonical progress monitor. ETA uses observed completed-task medians with hierarchical fallback by dataset/model/corruption, so the estimate is explicitly provisional until the task composition is well represented.
- `scripts/stop_full_study.ps1` requests a checkpoint after the current task is persisted. `scripts/stop_full_study.ps1 -Force` is an explicit human action for a stuck task; it may terminate the worker, but does not delete outputs or database rows.
- `scripts/resume_full_study.ps1 -ConfirmResume` requires the existing authorization artifact and task database, rejects an active worker, clears a prior stop request, launches in the background, and waits for a fresh heartbeat. Any row left `running` is returned to `pending`, increments `interruptions`, and does not consume a deterministic model-failure retry.
- A failed row remains failed after its configured attempts and blocks confirmatory analysis. A pending/retry/running row can never be silently treated as complete.
- The Phase 2 authorization also binds a runtime-code identity over the Python package, launcher/monitor scripts, and `pyproject.toml`; changing those files after authorization invalidates the authorization artifact.

## Authorization and outcome boundary

The full-study authorization is a separate artifact bound to the configuration hash, panel hash, split-manifest hash, task-manifest identity and file hash, Phase 1 `GO` decision, and Phase 1 validation artifact. `runnable` remains false until a human invokes `scripts/start_full_study.ps1 -ConfirmPreregisteredRun`. Preflight, split generation, task freezing, tests, and metadata verification do not access model outcomes.

## Runtime estimate

Data preparation is estimated at 10-30 minutes. Based on 126 completed Kill Test tasks (270.6 seconds of serial wall time) and the source pipeline's 10,920-task Phase 2 run (71,388 seconds), the expected serial single-worker execution time is **8-12 hours**. An optimistic range is 6-8 hours; a conservative range including large-dataset variance and deterministic retries is 16-30 hours. After launch, ETA is recalculated from the observed completed-task median and remaining task composition.

## Disclosures

OpenML 31, 44, and 1049 overlap the earlier missingness pipeline. Several other records overlap its discovery or confirmatory metadata panel. The Phase 1 overlap audit found direct outlier coverage in arXiv `2604.25154v1`; this study therefore claims a unified multi-corruption audit, not that TFM outlier robustness has never been studied.
