import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import numpy as np
import torch.nn.functional as F
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
    
    print(f"[{season_key.capitalize()} Weekday Router Attention]")
    
    # Weekday branches: 8 (Sin) and 9 (Cos)
    # We'll check Branch 8 (Sin) as a proxy
    branch = model.branches[8] 
    
    days = {
        "Mon (Idx 1)": 1,
        "Wed (Idx 3)": 3
    }
    
    for name, w in days.items():
        period = 7
        val = np.sin(2 * np.pi * w / period)
        inp = torch.full((1, config["seq_len"]), val, device=device).float()
        
        # Get Router Logits
        logits = branch.router(inp)
        probs = F.softmax(logits, dim=-1)
        
        # Mean across sequence length (they are all same day)
        avg_probs = probs.mean(dim=1).detach().numpy()[0]
        
        print(f"\n{name} - Router Probabilities:")
        for i, p in enumerate(avg_probs):
            print(f"  Expert {i}: {p:.4f}")
            
except Exception as e:
    print(f"Error: {e}")
