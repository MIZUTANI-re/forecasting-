import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import numpy as np
import sys
from pathlib import Path
from optimiser.models import MMKModel

# Setup
device = torch.device("cpu")

# Config (Must match training)
dim_in = 10 
n_experts_list = [4] * dim_in
config = {
    "seq_len": 72,
    "horizon": 24, 
    "exog_dim": dim_in - 1,
    "d_model": 128,
    "n_experts": n_experts_list,
    "grid_size": 3,
    "num_layers": 1
}

# Season Definitions
seasons = {
    'spring': [2, 3, 4],    # Mar, Apr, May
    'summer': [5, 6, 7],    # Jun, Jul, Aug
    'autumn': [8, 9, 10],   # Sep, Oct, Nov
    'winter': [11, 0, 1]    # Dec, Jan, Feb
}

print("=== Unified Month Contributions (Sin + Cos) ===")

for season_key, month_indices in seasons.items():
    model_path = Path(f"c:/Users/2213144/practice/learning_saver/MMK_{season_key}_separated.pth")
    
    if not model_path.exists():
        print(f"Skipping {season_key}: Model not found.")
        continue
        
    # Load Model
    model = MMKModel(**config).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as e:
        print(f"Error loading {season_key}: {e}")
        continue

    print(f"\n[{season_key.capitalize()}]")
    
    # Calculate Contribution for specific months
    for m in month_indices:
        # 0=Jan, 2=Mar... Month Number is m+1
        period = 12
        s = np.sin(2 * np.pi * m / period)
        c = np.cos(2 * np.pi * m / period)
        
        # Branches 6 (Sin) and 7 (Cos) for Month
        # Note: Input shape to branch is (B, Seq_Len)
        inp_s = torch.full((1, config["seq_len"]), s, device=device).float()
        out_s = model.branches[6](inp_s).mean().item()
        
        inp_c = torch.full((1, config["seq_len"]), c, device=device).float()
        out_c = model.branches[7](inp_c).mean().item()
        
        total = out_s + out_c
        print(f"  Month {m+1}: {total:+.4f}")
