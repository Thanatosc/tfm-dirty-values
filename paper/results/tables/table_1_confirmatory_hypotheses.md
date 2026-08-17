# Table 1. Preregistered confirmatory results

All effects are absolute ROC-AUC units. Confidence intervals are unadjusted 95% dataset-bootstrap intervals. Confirmatory decisions follow the two-sided exact sign-flip tests and the frozen Holm correction, not whether an unadjusted bootstrap interval excludes zero.

## H1: Mean TFM degradation at affected-cell rate 0.50

| Corruption | N datasets | Mean | 95% CI | Holm p | Decision |
|---|---:|---:|---:|---:|---|
| Measurement noise | 9 | 0.0525 | [0.0209, 0.0908] | 0.015625 | Positive TFM degradation |
| Outlier | 9 | 0.1001 | [0.0616, 0.1436] | 0.015625 | Positive TFM degradation |
| Quantization | 9 | 0.0290 | [0.0117, 0.0503] | 0.015625 | Positive TFM degradation |
| Unit error | 9 | 0.0908 | [0.0495, 0.1382] | 0.015625 | Positive TFM degradation |
| Categorical error | 7 | 0.0856 | [0.0135, 0.1923] | 0.015625 | Positive TFM degradation |

## H2: TFM minus tree-family degradation at rate 0.50

| Corruption | N datasets | Mean | 95% CI | Holm p | Decision |
|---|---:|---:|---:|---:|---|
| Measurement noise | 9 | 0.0061 | [-0.0056, 0.0197] | 0.843750 | Not significant; not evidence of equivalence |
| Outlier | 9 | -0.0152 | [-0.0390, 0.0090] | 0.808594 | Not significant; not evidence of equivalence |
| Quantization | 9 | 0.0012 | [-0.0052, 0.0081] | 0.843750 | Not significant; not evidence of equivalence |
| Unit error | 9 | 0.0422 | [0.0176, 0.0682] | 0.062500 | Not significant; not evidence of equivalence |
| Categorical error | 7 | -0.0173 | [-0.0288, -0.0065] | 0.062500 | Not significant; not evidence of equivalence |

## H3: Degradation slope over affected-cell rates

| Corruption | N datasets | Mean | 95% CI | Holm p | Decision |
|---|---:|---:|---:|---:|---|
| Measurement noise | 9 | 0.0897 | [0.0340, 0.1562] | 0.015625 | Positive dose-response slope |
| Outlier | 9 | 0.2131 | [0.1434, 0.2871] | 0.015625 | Positive dose-response slope |
| Quantization | 9 | 0.0508 | [0.0201, 0.0880] | 0.015625 | Positive dose-response slope |
| Unit error | 9 | 0.1008 | [0.0516, 0.1586] | 0.015625 | Positive dose-response slope |
| Categorical error | 7 | 0.1980 | [0.0352, 0.4441] | 0.031250 | Positive dose-response slope |

**Interpretive boundary.** No H2 contrast passes the preregistered alpha = 0.05 threshold. These non-detections must not be described as equivalence or as proof that the model families are equally robust.
