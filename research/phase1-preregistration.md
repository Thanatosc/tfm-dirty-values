# Phase 1 Kill Test Preregistration

Version: `phase1_prereg_v1`  
Frozen before dirty-value outcome access: `2026-08-14`

## Question

Does natural feature-value corruption produce a practically relevant and model-dependent loss in binary-classification ROC-AUC for a frozen TabPFN v2 compared with XGBoost?

## Design

The panel contains OpenML 31, 44, and 1049; seeds `1907`, `2027`, and `4049`; TabPFN v2 and XGBoost; and a stratified 70/30 split. Each dataset/model/seed has one clean baseline. Polluted tasks cross three corruption types with affected-cell rates 0.15 and 0.50.

Only observed numeric cells are eligible in Phase 1. The requested rate is applied separately to the context and query matrices, with the target count equal to `round(rate * eligible_observed_cells)`. All corruption statistics come from the unpolluted context split. Labels never enter injection.

## Frozen operators

| Type | Phase 1 operator | Fixed severity |
|---|---|---|
| Measurement noise | add independent Gaussian noise to selected cells | `sigma_noise / sigma_feature = 0.50` |
| Outlier | replace selected cells by `mean_feature +/- k * sigma_feature` | `k = 10` |
| Quantization | map selected values to centers of context-fitted equal-width bins | `8` bins |

The 0.15/0.50 axis is the affected-cell proportion, not the operator magnitude. This distinction prevents the overloaded word "intensity" from changing meaning between corruption types.

## Decision rules

For each dataset/model/corruption/rate, average `clean ROC-AUC - polluted ROC-AUC` across the three seeds. Datasets remain the inferential unit.

`GO` if either condition holds at rate 0.50:

1. For any corruption, the unweighted mean TabPFN degradation across datasets is at least 0.05 ROC-AUC.
2. For any corruption, the absolute unweighted mean of `(TabPFN degradation - XGBoost degradation)` is at least 0.02 ROC-AUC and all three dataset-level gaps have the same non-zero sign.

`STRESS_RETEST` if neither GO rule holds and every model/corruption dataset-average degradation at rate 0.50 is below 0.01 ROC-AUC. The one permitted retest uses sigma ratio 1.0, outlier `k=20`, and two-bin quantization at the same 0.50 affected-cell rate.

`NO_GO` if the stress retest also leaves every dataset-average degradation below 0.01 and the model-gap rule remains false.

`REVIEW` covers the otherwise ambiguous region: at least one degradation is 0.01 or larger, but neither GO rule holds. It requires a documented design decision; it is not an invitation to tune thresholds after inspecting results.

## Sanity and failure rules

- Any difference between repeated injection calls with identical inputs and seeds is a hard failure.
- Any mutation of caller-owned input arrays is a hard failure.
- Any selected categorical cell in C1-C4 or numerical cell in C5 is a hard failure.
- More than 5% failed task cells after one deterministic retry blocks interpretation.
- ECE, balanced accuracy, and log loss are diagnostic only in Phase 1.

