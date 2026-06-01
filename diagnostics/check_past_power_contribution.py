import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import numpy as np
from pathlib import Path
from optimiser.models import MMKModel

# Setup
device = torch.device("cpu")
season_key = 'spring'
model_path = Path(f"c:/Users/2213144/practice/learning_saver/MMK_{season_key}_separated.pth")

dim_in = 10 
config = {
    "seq_len": 72,
    "horizon": 24, 
    "exog_dim": dim_in - 1,
    "d_model": 128,
    "n_experts": [4] * dim_in,
    "grid_size": 3,
    "num_layers": 1
}

model = MMKModel(**config).to(device)
try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Sweep Past Power (normalized range usually -2 to 2 or -3 to 3)
    sweep = np.linspace(-3.0, 3.0, 100)
    contributions = []
    
    branch = model.branches[0] # Past Power is typically idx 0
    
    with torch.no_grad():
        for v in sweep:
            inp = torch.full((1, config["seq_len"]), v, device=device).float()
            out = branch(inp).mean().item()
            contributions.append(out)
            
    min_c = min(contributions)
    max_c = max(contributions)
    range_c = max_c - min_c
    
    print(f"[{season_key.capitalize()} Past Power Contributions]")
    print(f"Sweep Range: [-3.0, 3.0]")
    print(f"Min Contribution: {min_c:.4f}")
    print(f"Max Contribution: {max_c:.4f}")
    print(f"Range (Importance): {range_c:.4f}")
    
    # Check linearity
    slope = (contributions[-1] - contributions[0]) / (sweep[-1] - sweep[0])
    print(f"Approx Slope: {slope:.4f}")

except Exception as e:
    print(f"Error: {e}")
