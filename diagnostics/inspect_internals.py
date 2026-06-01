import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import glob
from pathlib import Path

def inspect_file(npz_path):
    print(f"\n{'='*60}")
    print(f"📄 File: {npz_path}")
    print(f"{'='*60}")
    
    data = np.load(npz_path)
    
    for key in data.files:
        if key == "feature_names":
            print(f"\n🔹 Feature Names: {list(data[key])}")
            continue
            
        val = data[key]
        avg_val = np.mean(val, axis=0) # Average over samples
        
        print(f"\n📊 Key: {key}")
        print(f"   Shape: {val.shape} -> Avg Shape: {avg_val.shape}")
        
        if avg_val.ndim == 0:
            print(f"   Value (Scalar/Avg): {avg_val:.4f}")
        elif avg_val.ndim == 1:
            print(f"   Values (Avg per Step):")
            print(avg_val)
        elif avg_val.ndim == 2:
            # (Horizon, D) or (N, N)
            row_mean = np.mean(avg_val, axis=1) # Average across feature/dim
            print(f"   Average across features (per time step):")
            print(row_mean)
            
            if avg_val.shape[0] <= 15 and avg_val.shape[1] <= 15:
                # Small enough to print fully (e.g. attention or small features)
                print(f"   Detailed Raw Values (Avg):")
                np.set_printoptions(precision=4, suppress=True)
                print(avg_val)
            else:
                print(f"   [Note] Tensor is too large ({avg_val.shape}) to print raw values. Use the generated PNGs for detailed patterns.")
        else:
             print(f"   Higher Dimensional Tensor: Shape {avg_val.shape}")

def main():
    patterns = [
        "result_csv/**/*internals.npz",
        "result_72_predict/**/*internals.npz"
    ]
    
    found_files = []
    for p in patterns:
        found_files.extend(glob.glob(p, recursive=True))
        
    if not found_files:
        print("❌ No internal state files found. Run a model in main.py first.")
        return
        
    for i, f in enumerate(found_files):
        print(f"{i+1}. {f}")
        
    choice = input("\nSelect a file number to inspect (or 'all'): ").strip()
    
    if choice.lower() == 'all':
        for f in found_files:
            inspect_file(f)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(found_files):
                inspect_file(found_files[idx])
            else:
                print("❌ Invalid index.")
        except ValueError:
            print("❌ Invalid input.")

if __name__ == "__main__":
    main()
