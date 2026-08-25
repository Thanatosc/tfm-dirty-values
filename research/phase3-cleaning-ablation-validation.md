# Phase 3 Cleaning Ablation Validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-25
- Verification Status: VERIFIED
- Version Label: phase3_validation_v1
- Preregistration: `research/phase3-cleaning-ablation-preregistration.md`
- Analysis source: `results/cleaning_ablation/analysis.json`
- Outcome boundary: interpretation is confined to the two preregistered confirmatory families

## Execution record

| Item | Value |
|---|---|
| Tasks completed | `2,425 / 2,425` |
| Failed | `0` |
| Retries consumed | `0` |
| Wall clock | `60.0` min (preregistered estimate `1.0-1.5` h) |
| Completion gate | `COMPLETE_PENDING_VALIDATION` |
| Result verification | `VERIFIED`, `0` errors, 2,425 JSON files matched against SQLite |
| Output set identity | `3809E92EF23FE60EFB209528...` |
| Arms executed | C `2,150`, D `275` |
| Arms reused read-only from Phase 2 | A `275`, B `550` at the two analysed rates |
| Paired-identity probe | `VERIFIED`, 5 of 5 cells, tolerance `0.0` |

Phase 2 was not modified. After Phase 3 completed, `generate_paper_results.py --check` still reports `14/14` paper artifacts matching the frozen source, and the Phase 2 authorization still validates against runtime identity `9325544b8adb82ea...`.

## Confirmatory result: H4, TFM ROC-AUC recovery

Family of ten, Holm-corrected. Positive means cleaning recovered discrimination.

| Mechanism | Rate | Phase 2 damage | Recovery | 95% CI | p Holm | n |
|---|---:|---:|---:|---|---:|---:|
| measurement noise | 0.15 | 0.0202 | −0.0054 | [−0.0115, −0.0003] | 0.6055 | 9 |
| measurement noise | 0.50 | 0.0525 | **−0.0124** | [−0.0213, −0.0039] | **0.0391** | 9 |
| outlier | 0.15 | 0.0317 | −0.0039 | [−0.0085, +0.0008] | 0.6055 | 9 |
| outlier | 0.50 | 0.1001 | −0.0039 | [−0.0116, +0.0002] | 0.7500 | 9 |
| quantization | 0.15 | 0.0101 | −0.0055 | [−0.0102, −0.0015] | 0.2812 | 9 |
| quantization | 0.50 | 0.0290 | −0.0075 | [−0.0142, −0.0020] | 0.2812 | 9 |
| unit error | 0.15 | 0.0558 | +0.0174 | [+0.0001, +0.0394] | 0.6055 | 9 |
| unit error | 0.50 | 0.0908 | +0.0042 | [−0.0089, +0.0171] | 0.7500 | 9 |
| categorical error | 0.15 | 0.0213 | −0.0021 | [−0.0038, −0.0006] | 0.2812 | 7 |
| categorical error | 0.50 | 0.0856 | −0.0030 | [−0.0060, −0.0005] | 0.2812 | 7 |

**One of ten tests crosses the corrected threshold, and it is negative.** Cleaning measurement-noise-corrupted data at rate 0.50 reduced TFM ROC-AUC by 0.0124. No mechanism showed a corrected positive recovery.

Unit error is the only mechanism with positive point estimates at both rates, and neither survives correction.

## Confirmatory result: H5, cleaning cost on already-clean data

TFM ROC-AUC change from applying the cleaner to clean input: mean **−0.0033**, 95% CI [−0.0066, −0.0005], p Holm `0.0781`, n = 11.

The point estimate is negative and the interval excludes zero, but the corrected exact test does not cross `0.05`. Per model, the cost is negative for all five: TabPFN −0.0033, TabICLv2 −0.0033, XGBoost −0.0065, LightGBM −0.0059, Random Forest −0.0047.

## Descriptive findings

**Direction is consistent across metrics.** For the four mechanisms with negative ROC-AUC recovery, balanced accuracy and log loss also degrade. Unit error improves on ROC-AUC, balanced accuracy, and log loss at rate 0.15.

**Unit error is the only model-family divergence.** TFM recovery is +0.0174 and +0.0042 while tree recovery is −0.0074 and −0.0117. This is descriptive; no confirmatory family-contrast test was preregistered for Phase 3.

**Residual damage exceeds original damage wherever recovery is negative**, which follows arithmetically from the identity recovery + residual = damage.

**Recovery-fraction exclusions behaved as designed.** The preregistered `0.005` minimum denominator excluded 4 of 7 dataset cells for categorical error at rate 0.15 and 3 of 9 for quantization at rate 0.15, where Phase 2 damage was too small for a stable ratio.

## Independent verification of the headline number

The significant H4 cell was recomputed from raw per-task SQLite records, bypassing `analysis.py` entirely: arm B matched to arm C on (dataset, seed, model, mechanism, rate), seeds averaged within dataset, then averaged across the nine datasets. Independent value **−0.012378** against the reported −0.0124. Arm B and arm C key sets were identical. The outlier cell at rate 0.15 was recomputed the same way: **−0.003942** against the reported −0.0039.

The per-cell arm ordering is visible directly in the raw values: clean(A) > dirty(B) > cleaned(C) in 16 of 18 model-dataset cells for measurement noise at rate 0.50.

## Interpretation, bounded by the preregistration

**What the ablation establishes.** A standard off-the-shelf cleaning pass, Tukey-fence winsorization plus unseen-category consolidation, does not recover TFM discrimination for four of the five dirty-value mechanisms, measurably harms it for measurement noise at the higher rate, and carries a consistently negative point-estimate cost when applied to data that was already clean.

**Why, mechanically.** Winsorization can only help when corruption pushes values outside the fence and clipping to the fence approximately restores the original. Unit errors of factor 1,000 satisfy that condition, and they are the one mechanism with positive recovery. Measurement noise at `sigma_ratio=0.50` stays largely inside the fence, so the clipped cells are genuine distribution tails rather than corrupted cells. Quantized values stay inside the data range for the same reason. For outliers the fence does clip the injected extremes at rate 0.15, yet ROC-AUC recovery is still slightly negative, which indicates the models were already substantially robust to 10σ outliers and the information lost by clipping genuine tails cost about as much as the outliers themselves.

**Registered expectations that were borne out.** Amendment 1 registered, before execution, that quantization would be poorly recoverable because binning destroys within-bin information irreversibly, and that mild in-distribution corruption could produce negative recovery because the cleaner clips real tails. Both occurred. The breakdown-point expectation for outliers at rate 0.50 also occurred and is why two rates were analysed.

**What the ablation does not establish.** Nine of ten non-detections are not evidence that cleaning is useless. This tested exactly one cleaner with one fixed parameterization on two rates, with nine numeric and seven categorical datasets as inferential units. A mechanism-aware cleaner, a different estimator with a higher breakdown point, a detection-then-repair pipeline, or imputation-based repair could behave differently. The result is a bounded negative finding about a standard default, not a general claim about data cleaning.

**The mechanism ranking must be reframed.** The preregistration described the ranking as a cleaning-priority ordering. Because only one mechanism has positive recovery and it does not survive correction, the honest deliverable is not a priority ordering but a **repair-attempt audit**: unit errors are the one mechanism where a standard cleaner plausibly helps, and for the rest a generic winsorization pass should not be expected to restore discrimination. The manuscript must not present the ordering as a validated cleaning benefit ranking.

## Statistical fallacy scan

| Fallacy | Status | Note |
|---|---|---|
| Treating non-significance as equivalence | CLEAR | Nine non-detections are reported as non-detections; the preregistration forbids equivalence language |
| p-hacking / rate selection after outcomes | CLEAR | Both rates and all ten tests were registered before execution; the rate widening was driven by a no-model synthetic probe |
| Multiple comparisons uncorrected | CLEAR | Holm within the family of ten for H4, family of one for H5 |
| Pseudo-replication on seeds | CLEAR | Seeds averaged within dataset; dataset is the inferential unit |
| Ratio instability near zero | CLEAR | Preregistered `0.005` minimum denominator; exclusions counted and reported |
| HARKing the mechanism explanation | CLEAR | The breakdown point and the tail-clipping cost were registered in Amendment 1 before execution |
| Baseline mismatch | CLEAR | Paired-identity probe `VERIFIED` at tolerance `0.0` on four metrics and both array hashes |
| Leakage via the cleaner | CLEAR | Fitted on corrupted context only; asserted by `test_cleaner_never_reads_the_query_partition` |
| Cleaning benefit confounded with transform benefit | CLEAR | Arm D isolates the cost of the cleaner on clean input |
| Selective metric reporting | CLEAR | All four metrics reported for every mechanism and rate |
| Overstating a single significant result | CLEAR | The one corrected result is negative and is reported as such, not as a cleaning recommendation |

## Reproducibility verdict

`VERIFIED`. Task identity matches the frozen manifest, all 2,425 outputs match SQLite byte-for-byte, the headline number reproduces from raw records through an independent code path, and Phase 2 evidence is unchanged.

## Deliverable status

The charter's promised third contribution changes character. It was specified as a cleaning-priority guide ranking mechanisms by AUC recovered per unit of cleaning effort. What the evidence supports is a bounded negative repair audit plus one candidate exception. The manuscript must state this plainly and must not restore the original framing.
