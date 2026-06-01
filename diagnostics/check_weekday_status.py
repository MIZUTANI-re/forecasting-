import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import numpy as np
import os
import datetime
from pathlib import Path
from optimiser.models import MMKModel

# 1. Check File Timestamp
file_path = Path(r"c:\Users\2213144\practice\picture_analysis\MMK_spring_Contribution_Weekday_Unified.png")
if file_path.exists():
    mtime = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
    print(f"File: {file_path.name}")
    print(f"Last Modified: {mtime}")
else:
    print(f"File not found: {file_path}")

# 2. Calculate Weekday Values
device = torch.device("cpu")
season_key = 'spring'
model_path = Path(f"c:/Users/2213144/practice/learning_saver/MMK_{season_key}_separated.pth")

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

model = MMKModel(**config).to(device)
try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"\n[{season_key.capitalize()} Weekday Contributions]")
    weekdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    for w in range(7):
        period = 7
        s = np.sin(2 * np.pi * w / period)
        c = np.cos(2 * np.pi * w / period)
        
        # Weekday Branches: 8 (Sin), 9 (Cos)
        inp_s = torch.full((1, config["seq_len"]), s, device=device).float()
        out_s = model.branches[8](inp_s).mean().item()
        
        inp_c = torch.full((1, config["seq_len"]), c, device=device).float()
        out_c = model.branches[9](inp_c).mean().item()
        
        total = out_s + out_c
        print(f"  {weekdays[w]} (Idx {w}): {total:+.4f}")

except Exception as e:
    print(f"Error loading model: {e}")
