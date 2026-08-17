# Table 2. Model-specific ROC-AUC degradation at affected-cell rate 0.50

Values are dataset-level means with unadjusted 95% dataset-bootstrap intervals.

| Corruption | Model | Family | N datasets | Mean | 95% CI |
|---|---|---|---:|---:|---:|
| Measurement noise | TabPFN v2 | TFM | 9 | 0.0540 | [0.0206, 0.0954] |
| Measurement noise | TabICL v2 | TFM | 9 | 0.0511 | [0.0210, 0.0874] |
| Measurement noise | XGBoost | Tree ensemble | 9 | 0.0503 | [0.0259, 0.0805] |
| Measurement noise | LightGBM | Tree ensemble | 9 | 0.0487 | [0.0253, 0.0785] |
| Measurement noise | Random Forest | Tree ensemble | 9 | 0.0401 | [0.0162, 0.0661] |
| Outlier | TabPFN v2 | TFM | 9 | 0.1005 | [0.0594, 0.1468] |
| Outlier | TabICL v2 | TFM | 9 | 0.0997 | [0.0632, 0.1397] |
| Outlier | XGBoost | Tree ensemble | 9 | 0.1199 | [0.0872, 0.1530] |
| Outlier | LightGBM | Tree ensemble | 9 | 0.1117 | [0.0795, 0.1431] |
| Outlier | Random Forest | Tree ensemble | 9 | 0.1144 | [0.0771, 0.1547] |
| Quantization | TabPFN v2 | TFM | 9 | 0.0265 | [0.0102, 0.0477] |
| Quantization | TabICL v2 | TFM | 9 | 0.0316 | [0.0124, 0.0552] |
| Quantization | XGBoost | Tree ensemble | 9 | 0.0285 | [0.0141, 0.0457] |
| Quantization | LightGBM | Tree ensemble | 9 | 0.0278 | [0.0128, 0.0454] |
| Quantization | Random Forest | Tree ensemble | 9 | 0.0273 | [0.0135, 0.0428] |
| Unit error | TabPFN v2 | TFM | 9 | 0.0884 | [0.0422, 0.1452] |
| Unit error | TabICL v2 | TFM | 9 | 0.0933 | [0.0539, 0.1368] |
| Unit error | XGBoost | Tree ensemble | 9 | 0.0499 | [0.0208, 0.0822] |
| Unit error | LightGBM | Tree ensemble | 9 | 0.0461 | [0.0199, 0.0767] |
| Unit error | Random Forest | Tree ensemble | 9 | 0.0497 | [0.0222, 0.0802] |
| Categorical error | TabPFN v2 | TFM | 7 | 0.0887 | [0.0164, 0.1934] |
| Categorical error | TabICL v2 | TFM | 7 | 0.0824 | [0.0097, 0.1918] |
| Categorical error | XGBoost | Tree ensemble | 7 | 0.1047 | [0.0258, 0.2229] |
| Categorical error | LightGBM | Tree ensemble | 7 | 0.1009 | [0.0222, 0.2171] |
| Categorical error | Random Forest | Tree ensemble | 7 | 0.1030 | [0.0199, 0.2202] |
