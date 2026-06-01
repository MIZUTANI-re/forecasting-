# Model Worst-Case Analysis

> **Note:** Analysis based on sliding window of prediction length (24) across aggregated results.

| Model | Worst Window Start | Max Window RMSE | True (Mean) | Pred (Mean) | Diff |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FiLM** | 2023-06-19 | **84.52** | 307.0 | 237.3 | -69.6 |
| **iTransformer** | 2023-08-14 | **95.05** | 279.8 | 363.7 | +83.9 |
| **TransformerFixed** | 2023-08-11 | **74.52** | 309.4 | 369.8 | +60.4 |
| **Hybrid_FiLM_Ablation** | 2023-08-14 | **58.01** | 280.2 | 328.2 | +47.9 |
| **Hybrid_Predict_Fusion** | 2024-02-12 | **59.80** | 295.9 | 343.2 | +47.3 |
| **Hybrid_NoHQ_Predict_Fusion** | 2023-08-14 | **76.54** | 282.7 | 334.2 | +51.6 |
| **Hybrid_Gated_FeatureFusion** | 2023-08-23 | **54.16** | 333.5 | 377.7 | +44.1 |
| **Hybrid_Gated_QueryFusion** | 2023-08-23 | **63.35** | 334.0 | 387.3 | +53.3 |
| **Hybrid_KAN_Gated_FeatureFusion** | 2023-08-16 | **59.85** | 320.6 | 270.8 | -49.8 |
| **htmformer** | 2023-08-11 | **79.30** | 310.8 | 370.1 | +59.4 |
| **MLP** | 2023-08-14 | **87.54** | 279.1 | 356.9 | +77.8 |
| **grid_tst** | 2023-08-21 | **75.44** | 384.5 | 321.7 | -62.7 |
| **MMK** | 2023-08-14 | **76.89** | 281.1 | 338.8 | +57.7 |
