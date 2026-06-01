# Standardized Model Benchmark Results

**Configuration:**
- **Epochs:** 50
- **Exclude COVID:** False (Off)
- **IEEJ Comparison:** False (Off)
- **Device:** CUDA (GPU)

## Summary Table (All Seasons)

| Model | RMSE | MAPE (%) | Parameters |
| :--- | :--- | :--- | :--- |
| **FiLM** | **25.46** | **5.64** | **16,544** |
| iTransformer | 16.35 | 3.56 | 277,400 |
| TransformerFixed | 16.48 | 3.66 | 669,057 |
| Hybrid_FiLM_Ablation | 14.99 | 3.32 | 924,121 |
| Hybrid_Predict_Fusion | 15.40 | 3.33 | 693,338 |
| Hybrid_NoHQ_Predict_Fusion | 15.42 | 3.36 | 702,880 |
| Hybrid_Gated_FeatureFusion | 14.95 | 3.27 | 541,658 |
| Hybrid_Gated_QueryFusion | 15.04 | 3.39 | 691,330 |
| Hybrid_KAN_Gated_FeatureFusion | 15.35 | 3.33 | 543,960 |
| htmformer | 16.33 | 3.66 | 39,283,544 |
| MLP | 16.16 | 3.61 | 43,224 |
| grid_tst | 24.26 | 5.47 | 2,008 |

> **Note:** Lower RMSE/MAPE is better.

---

## Detailed Results by Season

### Spring
| Model | RMSE | MAPE (%) |
| :--- | :--- | :--- |
| **FiLM** | 20.10 | 5.38 |
| iTransformer | 13.54 | 3.88 |
| TransformerFixed | 10.95 | 3.15 |
| Hybrid_FiLM_Ablation | 10.00 | 2.86 |
| Hybrid_Predict_Fusion | 10.45 | 2.86 |
| Hybrid_NoHQ_Predict_Fusion | 9.37 | 2.66 |
| Hybrid_Gated_FeatureFusion | 10.06 | 2.81 |
| Hybrid_Gated_QueryFusion | 10.00 | 2.84 |
| Hybrid_KAN_Gated_FeatureFusion | 9.97 | 2.73 |
| htmformer | 10.15 | 2.98 |
| transformer_oneshot | 9.11 | 2.71 |
| grid_tst | 19.43 | 5.58 |

### Summer
| Model | RMSE | MAPE (%) |
| :--- | :--- | :--- |
| **FiLM** | 27.54 | 5.69 |
| iTransformer | 20.86 | 4.13 |
| TransformerFixed | 20.64 | 4.32 |
| Hybrid_FiLM_Ablation | 18.81 | 3.84 |
| Hybrid_Predict_Fusion | 19.24 | 3.88 |
| Hybrid_NoHQ_Predict_Fusion | 17.86 | 3.70 |
| Hybrid_Gated_FeatureFusion | 19.27 | 3.90 |
| Hybrid_Gated_QueryFusion | 19.76 | 4.20 |
| Hybrid_KAN_Gated_FeatureFusion | 19.15 | 3.78 |
| htmformer | 18.50 | 3.81 |
| transformer_oneshot | 18.82 | 3.71 |
| grid_tst | 25.77 | 5.46 |

### Autumn
| Model | RMSE | MAPE (%) |
| :--- | :--- | :--- |
| **FiLM** | 23.05 | 5.62 |
| iTransformer | 13.53 | 3.32 |
| TransformerFixed | 13.73 | 3.54 |
| Hybrid_FiLM_Ablation | 11.93 | 2.98 |
| Hybrid_Predict_Fusion | 12.24 | 3.20 |
| Hybrid_NoHQ_Predict_Fusion | 12.88 | 3.18 |
| Hybrid_Gated_FeatureFusion | 11.75 | 2.98 |
| Hybrid_Gated_QueryFusion | 12.51 | 3.30 |
| Hybrid_KAN_Gated_FeatureFusion | 11.60 | 2.98 |
| htmformer | 14.09 | 3.63 |
| transformer_oneshot | 11.94 | 3.03 |
| grid_tst | 22.89 | 6.10 |

### Winter
| Model | RMSE | MAPE (%) |
| :--- | :--- | :--- |
| **FiLM** | 26.39 | 5.97 |
| iTransformer | 20.57 | 4.83 |
| TransformerFixed | 19.87 | 4.89 |
| Hybrid_FiLM_Ablation | 18.11 | 4.32 |
| Hybrid_Predict_Fusion | 18.29 | 4.35 |
| Hybrid_NoHQ_Predict_Fusion | 17.25 | 4.09 |
| Hybrid_Gated_FeatureFusion | 17.65 | 4.21 |
| Hybrid_Gated_QueryFusion | 18.48 | 4.33 |
| Hybrid_KAN_Gated_FeatureFusion | 17.66 | 4.08 |
| htmformer | 19.22 | 4.86 |
| transformer_oneshot | 17.77 | 4.22 |
| grid_tst | 25.69 | 6.21 |

