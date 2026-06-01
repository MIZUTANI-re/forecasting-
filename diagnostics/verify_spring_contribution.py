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
season_key = 'spring'
model_path = Path(f"c:/Users/2213144/practice/learning_saver/MMK_{season_key}_separated.pth")

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

# Load Model
model = MMKModel(**config).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Helper calc
def get_month_contribution(month_idx):
    # month_idx: 0=Jan, 2=Mar...
    period = 12
    s = np.sin(2 * np.pi * month_idx / period)
    c = np.cos(2 * np.pi * month_idx / period)
    
    # Branches 6 and 7 are Month Sin/Cos
    inp_s = torch.full((1, config["seq_len"]), s, device=device).float()
    out_s = model.branches[6](inp_s).mean().item()
    
    inp_c = torch.full((1, config["seq_len"]), c, device=device).float()
    out_c = model.branches[7](inp_c).mean().item()
    
    return out_s + out_c

# Check Spring Months (Mar, Apr, May -> Indices 2, 3, 4)
print(f"Spring Month Contributions:")
for m in [2, 3, 4]:
    val = get_month_contribution(m)
    print(f"Month Index {m} (Month {m+1}): {val:.4f}")
