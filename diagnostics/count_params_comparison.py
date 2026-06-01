import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import sys
from pathlib import Path
from optimiser.models import MMKModel, iTransformer

# Config based on experiment_summary.csv (Spring)
config = {
    "seq_len": 72,
    "pred_len": 24, # horizon
    "in_dim": 9,    # exog_dim (10 variables total = 1 target + 9 exog)
    "d_model": 128,
    "dropout": 0.1,
    "num_layers_it": 2, # iTransformer
    "num_layers_mmk": 1, # MMK (1 layer as confirmed)
    "n_experts": 4,
    "grid_size": 3,
    "nhead": 4,
    "hidden_dim": 256
}

# 1. MMK Model
# Note: MMKModel init uses 'horizon' and 'exog_dim'
mmk = MMKModel(
    seq_len=config["seq_len"],
    exog_dim=config["in_dim"],
    horizon=config["pred_len"],
    d_model=config["d_model"],
    n_experts=config["n_experts"],
    grid_size=config["grid_size"],
    num_layers=config["num_layers_mmk"]
)
mmk_params = sum(p.numel() for p in mmk.parameters())

# 2. iTransformer Model
# Note: iTransformer init uses 'pred_len' and 'in_dim' (total vars including target? usually in_dim is total)
# iTransformer "in_dim" usually means number of variates. 
# In MMK "exog_dim"=9 means 10 vars total.
# So iTransformer in_dim should be 10.
it = iTransformer(
    seq_len=config["seq_len"],
    pred_len=config["pred_len"],
    in_dim=config["in_dim"] + 1, # 10 variables
    d_model=config["d_model"],
    nhead=config["nhead"],
    num_layers=config["num_layers_it"],
    hidden_dim=config["hidden_dim"],
    dropout=config["dropout"]
)
it_params = sum(p.numel() for p in it.parameters())

print(f"MMK (1 Layer, 4 Experts): {mmk_params:,} parameters")
print(f"iTransformer (2 Layers): {it_params:,} parameters")
