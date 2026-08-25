# Phase 3 Cleaning Ablation Preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-25
- Verification Status: UNVERIFIED
- Version Label: phase3_prereg_v1
- Outcome boundary: no Phase 3 model outcome was accessed when this document was frozen
- Authorization basis: project owner authorized unfreezing on 2026-08-25 for this ablation only

## Why this ablation exists

The project charter (`题目A-TFM特征污染审计-立项材料.md` §8, risk register) pre-authorizes exactly one contingency experiment:

> 审稿人要求提出修复方法 → 以"评测研究"定位抵御;清洗优先级表即为实践贡献。**可补一个最小消融(清洗 vs 不清洗的收益)**,不扩为方法章

Phase 2 delivered a bounded risk map but explicitly disclaimed a cleaning cost-benefit guide, and its Limitations state that no repair intervention was tested. Neurocomputing desk-rejected the manuscript on 2026-08-25 for weak distinctive contribution. This ablation restores the charter's third promised contribution by measuring what a standard, off-the-shelf cleaning pass recovers.

It is deliberately minimal. It does **not** propose a novel repair algorithm, tune a cleaner, compare cleaning methods, or extend into a methods chapter. One fixed cleaner is applied uniformly.

## Relationship to frozen Phase 2 evidence

Phase 2 is **not re-run and not modified**. Its 4,575 results, `analysis.json`, SQLite database, task manifest, splits, and authorization artifact remain byte-frozen and are reused as the comparison arms.

To keep the Phase 2 `runtime_code_identity_sha256` (`9325544b8adb82eaa4ddbc99c12b01364315edeaddf17e7fe68d7e41efe8700d`) valid, all Phase 3 code lives in the sibling package `src/tfm_dirty_values_ablation/`, which is outside the hashed Phase 2 runtime surface. Phase 3 imports the frozen `tfm_dirty_values` package for data loading, splits, corruption injection, model adapters, and metrics. No file inside `src/tfm_dirty_values/` is edited.

## Design

### Arms

Four arms per dataset-seed-model cell. Two are reused from frozen Phase 2 results; two are executed in Phase 3.

| Arm | Corruption | Cleaning | Source |
|---|---|---|---|
| A. clean baseline | none, rate 0 | no | **reused** from frozen Phase 2 |
| B. corrupted, uncleaned | mechanism at rate 0.50 | no | **reused** from frozen Phase 2 |
| C. corrupted, cleaned | mechanism at rate 0.50 | yes | **executed** in Phase 3 |
| D. clean, cleaned | none, rate 0 | yes | **executed** in Phase 3 |

Arm D is a mandatory cost control. Without it, a gain in arm C cannot be separated from the cleaner acting as a generally beneficial transform rather than a repair. Arm D also quantifies the penalty of cleaning data that did not need it, which is the practical cost side of any cleaning recommendation.

### Estimands

For each corruption mechanism, with the dataset as the independent unit and the five seeds averaged arithmetically before any test:

- **Recovery** = `ROC-AUC(C) − ROC-AUC(B)`. Positive means cleaning recovered discrimination lost to corruption.
- **Residual damage** = `ROC-AUC(A) − ROC-AUC(C)`. How much loss survives cleaning.
- **Recovery fraction** = `Recovery / (ROC-AUC(A) − ROC-AUC(B))`, reported descriptively only, and only for dataset cells whose Phase 2 denominator exceeds `0.005` ROC-AUC. Cells at or below that threshold are excluded from the fraction and the exclusion count is reported. The ratio is unstable near zero denominators and is never used for confirmatory inference.
- **Cleaning cost on clean data** = `ROC-AUC(D) − ROC-AUC(A)`. Negative means the cleaner harmed already-clean data.

### Confirmatory hypotheses

Same inference machinery as Phase 2: two-sided exact sign-flip tests on dataset-level aggregates, `alpha = 0.05`, Holm correction within each hypothesis family, 10,000 dataset bootstrap resamples for intervals.

- **H4** (family of five, one per mechanism): mean TFM recovery differs from zero. TFM mean is the average of TabPFN v2 and TabICLv2, matching the H1 construction.
- **H5** (single test, its own family): mean TFM cleaning cost on clean data differs from zero.

Tree-model recovery, per-model contrasts, secondary metrics, and the mechanism ranking are **descriptive with intervals**, not confirmatory. This keeps the multiple-comparison family small enough to be interpretable, following the Phase 2 precedent that individual model contrasts stay exploratory.

### Preregistered interpretation rules

- A non-significant H4 for a mechanism means no detected recovery. It does **not** establish that cleaning is useless for that mechanism.
- Quantization is expected a priori to be poorly recoverable because binning destroys within-bin information irreversibly. Recording this expectation before execution prevents a post hoc explanation from being presented as a finding.
- The mechanism ranking by recovery is the deliverable "cleaning priority" ordering. It ranks by measured ROC-AUC recovery under this one cleaner. It is **not** a cost-benefit guide: it carries no prevalence, detection-accuracy, repair-cost, or downstream-loss quantity, and the manuscript must continue to say so.
- No claim about natural dirty data follows. All corruption remains synthetic and single-mechanism.

## The cleaner

One fixed, mechanism-agnostic pipeline. Deliberately textbook, so that no reviewer can read it as a tuned contribution.

**Numeric columns** — robust winsorization at Tukey fences. Bounds are `median ± 1.5 × IQR` computed per column. Values outside are clipped to the fence.

**Categorical columns** — unseen-level consolidation. Any level in the query partition absent from the fitted context level set is mapped to the fitted context mode of that column.

### Leakage boundary, mandatory

- The cleaner is fitted **only on the corrupted context partition** of the same task. It never sees clean data, never sees the query partition, and never sees labels.
- Fitted bounds are then applied to both the context and the query partition, matching the dual-partition contamination design of Phase 2.
- This mirrors deployment: a practitioner holds dirty training data, estimates cleaning parameters from it, and applies them to incoming rows.
- Arm D fits the same cleaner on the clean context, so arms C and D differ only in whether the input was corrupted.

### Fixed parameters

| Parameter | Value | Rationale |
|---|---|---|
| Numeric method | Tukey fence winsorization | standard robust outlier treatment |
| IQR multiplier | `1.5` | conventional default, not tuned |
| Categorical method | unseen level to context mode | standard unseen-category handling |
| Applied to | all numeric columns / all categorical columns | no per-mechanism selection |

The same cleaner runs for all five mechanisms. No mechanism-specific variant, no parameter search, no selection based on observed outcomes.

## Scope and frozen task count

Two affected-cell rates: `0.15` and the Phase 2 primary rate `0.50`. Rates `0.05` and `0.30` are out of scope. There is no cleaning dose-response curve claim; two rates are the minimum needed to separate a genuine non-recovery from a robust-estimator breakdown artifact, for the reason recorded below.

- Arm C: `4 numeric mechanisms × 9 datasets × 5 seeds × 5 models × 2 rates = 1,800`, plus `categorical_error × 7 datasets × 5 seeds × 5 models × 2 rates = 350`, totalling `2,150`.
- Arm D: `11 datasets × 5 seeds × 5 models = 275`. Arm D has no rate dimension; the input is clean by definition.
- **Total Phase 3 tasks: 2,425.**

Panels, splits, seeds, models, checkpoints, and corruption parameters are inherited unchanged from `configs/full_study.yaml`.

### Amendment 1: why two rates, registered before any Phase 3 outcome

This preregistration was first frozen with a single rate of `0.50`. A no-model, synthetic-array probe of the cleaner (no dataset, no checkpoint, no metric) then revealed a structural problem with that scope.

The Tukey fence has a breakdown point of roughly 25 percent. At an affected-cell rate of `0.50`, injected outliers are half of the observed cells, so they capture the median and inflate the IQR that define the fence. The fence widens to contain the contamination and clips nothing. In the probe, the outlier mechanism changed `0.0` percent of context cells at rate `0.50` while changing `7.9` to `30.1` percent at rates `0.05` through `0.30`.

A rate-`0.50`-only design would therefore have reported that standard cleaning cannot repair outlier corruption. That statement would be an artifact of the estimator's breakdown point, not a property of cleaning, and outlier corruption is the mechanism that triggered the Phase 1 `GO` decision. Reporting it would have been a measurement error presented as a result.

Rate `0.15` sits below the breakdown point and rate `0.50` sits above it. The pair makes the mechanism explicit and testable instead of hidden. Rate `0.30` was not added because it sits near the boundary and would raise the task count without separating the two regimes any further.

The probe used synthetic Gaussian arrays and measured mean absolute cell distance to the uncorrupted matrix. It produced no ROC-AUC, touched no dataset in the panel, and loaded no model checkpoint. It is recorded here as design evidence, not as a result. Directions observed on synthetic cells need not carry over to ROC-AUC on the real panel.

### Amendment 1 consequences for the hypotheses

- **H4** becomes a family of ten: five mechanisms at two rates. Holm correction applies across all ten within the H4 family.
- **H5** is unchanged, since arm D has no rate dimension.
- The breakdown-point expectation is now preregistered: for mechanisms whose corruption is symmetric and large in magnitude, recovery at rate `0.50` may be near zero **because the cleaner's own parameters are corrupted**. Where that occurs, the manuscript must attribute it to the estimator's breakdown point and must not report it as evidence that the damage is intrinsically unrepairable.
- The preregistered quantization expectation is unchanged and is independent of the breakdown point: binning destroys within-bin information irreversibly, so clipping cannot restore it at any rate.
- A newly anticipated outcome is registered here: for mild in-distribution corruption such as measurement noise, and for quantization, winsorization may produce **negative** recovery, because the cells it clips are genuine distribution tails rather than corrupted cells. If observed, this is a real cost of applying a standard cleaner to data that does not need it, and it is the same quantity arm D measures on clean input.

## Runtime estimate

From the completed Phase 2 SQLite record: 4,575 tasks in 1.96 h wall clock, mean 1.54 s, median 1.17 s per task; the rate-0.50 subset of 1,075 tasks took 0.47 h. Per-model means were TabPFN 3.59 s, TabICLv2 2.09 s, Random Forest 1.47 s, XGBoost 0.31 s, LightGBM 0.27 s.

Phase 3 executes 2,425 tasks of comparable composition plus a cleaning step that is O(rows × columns) and negligible against model inference. Scaling the Phase 2 mean of 1.54 s per task gives roughly 1.04 h. Estimated wall clock **1.0–1.5 h**, conservatively **up to 3 h** including data loading and process startup.

## Execution gates

- Phase 3 uses its own config `configs/cleaning_ablation.yaml`, its own result directory `results/cleaning_ablation/`, its own SQLite database, and its own hash-bound authorization artifact. It cannot write into `results/full_study/`.
- The Phase 2 authorization artifact, database, and outputs are treated as read-only inputs.
- A paired-identity precondition must pass before any Phase 3 outcome is produced: for a preregistered probe set of task cells, Phase 3 must reproduce the frozen Phase 2 arm-B metrics exactly from the same split and corruption seeds. This proves the comparison is genuinely paired rather than merely similar. Any mismatch blocks execution.
- Confirmatory analysis is refused unless all 2,425 tasks are complete with zero failures.

### Paired-identity probe result and the defect it caught

The probe ran before any Phase 3 outcome existed. Its first execution returned `BLOCKED` with 3 of 5 cells matching. Quantization and categorical errors mismatched on all four metrics **and** on the corrupted-array SHA-256 hashes, which localized the fault to corruption injection rather than to the model or the metric.

Root cause: `tfm_dirty_values.injection._stable_seed` derives the corruption RNG seed by string-formatting the parameters dictionary, so the dictionary's insertion order changes the seed, the affected-cell mask, and the corrupted array. Phase 2 executed with **key-sorted** parameters, because its task manifest was serialized through `canonical_json`, persisted to SQLite as `parameters_json`, and rebuilt with `json.loads`. The first Phase 3 implementation read the same values directly from YAML, preserving YAML order.

Only mechanisms with more than one parameter key can be affected. Quantization (`method`, `bins`) and categorical errors (`strategy`, `novel_level_fraction`) diverged; measurement noise (`sigma_ratio`), outliers (`k`), and unit errors (`factors`) are single-key and matched. That pattern is exactly what the observed 3-of-5 result showed.

The defect was in new Phase 3 code. No file inside `src/tfm_dirty_values/` was modified, and no frozen Phase 2 result was touched. The fix routes every Phase 3 parameter through `tfm_dirty_values_ablation.design.normalize_parameters`, which returns key-sorted order. After the fix the probe returned `VERIFIED` with 5 of 5 cells matching at tolerance `0.0` on `roc_auc`, `balanced_accuracy`, `log_loss`, `ece_10_equal_width`, and both `model_X_train_sha256` and `model_X_test_sha256`.

This mattered beyond the probe. Had the probe not been required, arm C would have corrupted two of the five mechanisms differently from frozen arm B, and every recovery estimate for quantization and categorical errors would have been computed against a mismatched baseline. The recorded probe report is `artifacts/phase3/paired-identity-probe.json`, and the behaviour is pinned by `tests/test_ablation_design.py::test_parameter_key_order_changes_the_corruption_seed`.

## Deliverable into the manuscript

One new subsection in Results and one row-per-mechanism table: recovery, residual damage, cleaning cost on clean data, with intervals and corrected p-values. One added paragraph in Practical Implications giving the priority ordering. Discussion and Limitations are updated to state that a single standard cleaner was tested and that repair-cost quantities remain absent.

The manuscript's existing claims are not revised. H1, H2, H3, and RQ4 results stand unchanged; this ablation adds an arm, it does not reinterpret Phase 2.
