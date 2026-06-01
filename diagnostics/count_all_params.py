import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import sys
from pathlib import Path
import torch
import pandas as pd

# Add project root


from optimiser.models import get_model
from optimiser.run_experiment import HP, SEQ_LEN, HORIZON

# Define models to check
MODELS_TO_CHECK = [
    "iTransformer",
    "grid_tst", 
    "Hybrid_Gated_FeatureFusion",
    "Hybrid_Gated_QueryFusion", 
    "Hybrid_KAN_Gated_FeatureFusion", 
    "Hybrid_FiLM_Ablation",
    "Hybrid_Predict_Fusion",
    "Hybrid_NoHQ_Predict_Fusion",
    "htmformer",
    "transformer_oneshot",
    "TransformerFixed",
    "MMK"
]

def count_all_models():
    results = []
    with open("params_full.txt", "w", encoding="utf-8") as f:
        f.write(f"{'Model Name':<35} | {'Total Params':<15} | {'Trainable':<15}\n")
        f.write("-" * 70 + "\n")
        
        in_dim = 9
        
        for model_name in MODELS_TO_CHECK:
            try:
                # Prepare Config
                config = HP.copy()
                config.update({
                    "seq_len": SEQ_LEN,
                    "pred_len": HORIZON,
                    "in_dim": in_dim,
                    "film_mode": "bidir", # Default
                    "use_hq": True,       # Default
                    # HTMformer specific
                    "nhead_var": HP["nhead"],
                })
                
                # MMK Constraint
                if model_name == "MMK":
                     # Use the Lightweight configuration we saw earlier
                     config["n_experts"] = [4] * (in_dim + 1) # Simple assumption
                     config["grid_size"] = 3
                     config["num_layers"] = 1 # MMK usually shallower or different

                # Initialize
                model = get_model(model_name, config)
                
                if model is None:
                    line = f"{model_name:<35} | {'FAILED':<15} | {'-':<15}\n"
                    print(line.strip())
                    f.write(line)
                    continue
                    
                total = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                line = f"{model_name:<35} | {total:<15,} | {trainable:<15,}\n"
                print(line.strip())
                f.write(line)
                results.append({"Model": model_name, "Parameters": total})
                
            except Exception as e:
                line = f"{model_name:<35} | {'Error':<15} | {str(e)}\n"
                print(line.strip())
                f.write(line)

if __name__ == "__main__":
    count_all_models()
