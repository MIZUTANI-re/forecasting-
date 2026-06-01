import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import sys
from pathlib import Path
import torch

# Add project root


from optimiser.models import MMKModel

def count_parameters():
    print("Initializing MMK Model to count parameters...")
    # Config matching the training (from experiment_summary or main.py)
    dim_in = 10
    config = {
        "seq_len": 72,
        "pred_len": 24, 
        "in_dim": dim_in - 1, # 9
        "d_model": 128,
        "n_experts": [4] * dim_in, # List of 4 experts per feature
        "grid_size": 3,
        "num_layers": 1 # Likely 1 based on previous context, or check main.py default
    }
    
    # Check experiment summary for overrides? 
    # The summary says: d_model=128, num_layers=2 (Wait, let me double check the summary)
    # Summary line: MMK,all,power_demand,17.08...,72,24,24,True,bidir,128,4,2,...
    # "2" is num_layers in the CSV header? 
    # CSV Header: ...d_model,nhead,num_layers...
    # Value: ...128,4,2...
    # So num_layers is 2.
    
    config["num_layers"] = 2
    
    model = MMKModel(
        seq_len=config["seq_len"],
        exog_dim=config["in_dim"],
        horizon=config["pred_len"],
        d_model=config["d_model"],
        n_experts=config["n_experts"],
        grid_size=config["grid_size"],
        num_layers=config["num_layers"]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

if __name__ == "__main__":
    count_parameters()
