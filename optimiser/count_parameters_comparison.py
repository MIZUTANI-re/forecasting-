
import torch
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from optimiser.models import get_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_mmk_lightweight_config():
    # Simulate the feature names we would have with use_onehot=False
    # [past_demand, temperature, precipitation, solar_mean, weekday, month, restday, holiday_onehot]
    feature_names = ["power_demand", "temperature", "precipitation", "solar_mean", "weekday", "month", "restday", "holiday_onehot"]
    exog_dim = len(feature_names) - 1
    
    n_experts_list = []
    # 1. Past demand
    n_experts_list.append(4)
    # 2. Exog
    for feat in feature_names[1:]:
        if any(k in feat for k in ["weekday", "month", "holiday", "restday"]):
            n_experts_list.append(1)
        else:
            n_experts_list.append(4)
            
    return {
        "seq_len": 72,
        "pred_len": 24,
        "in_dim": exog_dim,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 1,
        "hidden_dim": 256,
        "dropout": 0.1,
        "n_experts": n_experts_list,
        "grid_size": 3
    }

def get_standard_config(model_name):
    return {
        "seq_len": 72,
        "pred_len": 24,
        "in_dim": 15 if model_name != "MMK" else 15, # Standard exog dim with one-hot
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "hidden_dim": 256,
        "dropout": 0.1,
        "n_experts": 4,
        "grid_size": 3
    }

def compare():
    print("=" * 60)
    print(f"{'Model Profile':<35} | {'Parameters':<15}")
    print("=" * 60)
    
    # 1. MMK Original (Simulated)
    mmk_orig = get_model("MMK", {**get_standard_config("MMK"), "num_layers": 1})
    params_orig = count_parameters(mmk_orig)
    print(f"{'MMK (Original, 64 Experts)':<35} | {params_orig:,}")
    
    # 2. MMK Lightweight
    mmk_lite = get_model("MMK", get_mmk_lightweight_config())
    params_lite = count_parameters(mmk_lite)
    reduction = (1 - params_lite / params_orig) * 100
    print(f"{'MMK (Lightweight, 8 Branches)':<35} | {params_lite:,} ({reduction:.1f}% reduction)")
    
    # 3. iTransformer
    itrans = get_model("iTransformer", get_standard_config("iTransformer"))
    params_itrans = count_parameters(itrans)
    print(f"{'iTransformer (Standard)':<35} | {params_itrans:,}")

    # 4. Hybrid_Predict_Fusion
    hybrid_pred = get_model("Hybrid_Predict_Fusion", get_standard_config("Hybrid_Predict_Fusion"))
    params_hybrid_pred = count_parameters(hybrid_pred)
    print(f"{'Hybrid_Predict_Fusion (Standard)':<35} | {params_hybrid_pred:,}")

    # 5. Hybrid_FiLM_Ablation
    hybrid_film = get_model("Hybrid_FiLM_Ablation", get_standard_config("Hybrid_FiLM_Ablation"))
    params_hybrid_film = count_parameters(hybrid_film)
    print(f"{'Hybrid_FiLM_Ablation (Standard)':<35} | {params_hybrid_film:,}")

    # 6. TransformerFixed
    trans_fixed = get_model("TransformerFixed", get_standard_config("TransformerFixed"))
    params_trans_fixed = count_parameters(trans_fixed)
    print(f"{'TransformerFixed (Standard)':<35} | {params_trans_fixed:,}")
    
    print("-" * 60)
    print(f"MMK (Lightweight) is now {params_lite / params_itrans:.2f}x the size of iTransformer.")
    print("=" * 60)

if __name__ == "__main__":
    compare()
