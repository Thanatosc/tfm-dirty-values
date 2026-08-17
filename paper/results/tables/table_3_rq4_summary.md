# Table 3. Secondary-metric direction and interval summary at rate 0.50

Each cell reports the number of model-specific dataset-bootstrap intervals wholly above zero out of five models. The ECE column also reports the number of positive means because its direction was less consistent.

| Corruption | ROC-AUC CI > 0 | Balanced accuracy CI > 0 | Log loss CI > 0 | ECE positive mean | ECE CI > 0 |
|---|---:|---:|---:|---:|---:|
| Measurement noise | 5/5 | 5/5 | 4/5 | 4/5 | 3/5 |
| Outlier | 5/5 | 5/5 | 5/5 | 4/5 | 4/5 |
| Quantization | 5/5 | 5/5 | 4/5 | 4/5 | 2/5 |
| Unit error | 5/5 | 5/5 | 4/5 | 4/5 | 3/5 |
| Categorical error | 5/5 | 5/5 | 4/5 | 4/5 | 3/5 |

**Descriptive onset diagnostic.** ROC-AUC intervals first became wholly positive at an earlier tested rate in 19/25 model-corruption combinations; both metrics first crossed at the same tested rate in 6/25; ECE was earlier in 0/25. This rule was descriptive, not a multiple-comparison-adjusted confirmatory test.
