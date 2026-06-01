import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from optimiser.models import MMKModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calc_hybrid():
    print("Pre-computation of Feature Sets:")
    
    # --- New Proposal: Cyclic (10 vars) + 4 Experts ---
    # Features: Target(1) + Cont(3) + Cyclic/Bool(6) = 10
    dim_new = 10
    # User wants "4 brains" for combined things
    n_experts_unified = [4] * dim_new
    
    model_hybrid = MMKModel(
        seq_len=72,
        exog_dim=dim_new - 1, 
        horizon=24,
        d_model=128,
        n_experts=n_experts_unified,
        grid_size=3,
        num_layers=1
    )
    p_hybrid = count_parameters(model_hybrid)
    
    # Comparison points
    p_orig = 778816 # From previous run
    p_lite = 243380 # From previous run
    
    print("-" * 40)
    print(f"Proposed MMK (Cyclic, 4 Experts):")
    print(f"  - Input Features: {dim_new}")
    print(f"  - Experts per Branch: 4")
    print(f"  - Total Parameters: {p_hybrid:,}")
    print("-" * 40)
    
    red_orig = (1 - p_hybrid / p_orig) * 100
    inc_lite = (p_hybrid / p_lite)
    
    print(f"vs Original (OneHot): {red_orig:.1f}% Reduction")
    print(f"vs Lite (1 Expert):   {inc_lite:.1f}x Size")

if __name__ == "__main__":
    calc_hybrid()
