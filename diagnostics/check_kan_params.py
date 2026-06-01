import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
from optimiser.models import KANLinear

layer = KANLinear(72, 24, grid_size=3, spline_order=3)
params = sum(p.numel() for p in layer.parameters())
print(f"KANLinear(72, 24, grid=3, order=3) params: {params}")

# Detailed breakdown
print("Shapes:")
for name, p in layer.named_parameters():
    print(f"  {name}: {p.shape} ({p.numel()})")
