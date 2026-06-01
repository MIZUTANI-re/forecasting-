import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import sys
import os
import inspect

# Add path to sys.path
sys.path.append(r"C:\Users\2213144\practice")

from optimiser.models import TransformerEncoderDecoderOneShot

print("=== INSPECTING TransformerEncoderDecoderOneShot ===")
print("File:", inspect.getfile(TransformerEncoderDecoderOneShot))
print("Signature:", inspect.signature(TransformerEncoderDecoderOneShot.__init__))
