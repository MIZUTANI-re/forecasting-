import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import glob
from pathlib import Path

def print_film_stats(npz_path):
    data = np.load(npz_path)
    model_name = Path(npz_path).name.replace("_internals.npz", "")
    
    print(f"\n{'='*80}")
    print(f"📈 FiLM Coefficient Analysis: {model_name}")
    print(f"{'='*80}")
    
    # 1. Check for alpha (if it's a Gated model)
    for alpha_key in ["alpha", "gate_alpha"]:
        if alpha_key in data:
            alpha = np.mean(data[alpha_key], axis=0).flatten()
            print(f"\n🔹 Gating Weights ({alpha_key}) - Average over dataset:")
            print("Step | Weight (0:Variable, 1:Temporal)")
            print("-" * 35)
            for i, val in enumerate(alpha):
                print(f"{i+1:4} | {val:.4f}")
    
    # 2. Check for FiLM Gamma/Beta
    found_film = False
    for group in ["tv", "vt"]:
        gamma_key = f"gamma_{group}"
        beta_key = f"beta_{group}"
        
        if gamma_key in data and beta_key in data:
            found_film = True
            gamma = np.mean(data[gamma_key], axis=0) # (Horizon, D)
            beta = np.mean(data[beta_key], axis=0)   # (Horizon, D)
            
            # Since D is large (e.g. 128), we show the magnitude (mean absolute value)
            # to see which time steps are most active
            gamma_mag = np.mean(np.abs(gamma), axis=1)
            beta_mag = np.mean(np.abs(beta), axis=1)
            
            print(f"\n🔥 FiLM {group.upper()} (Temporal -> Variable if TV, Variable -> Temporal if VT):")
            print("Step | Avg |Gamma| (Sensitivity) | Avg |Beta| (Bias Shift)")
            print("-" * 55)
            for i in range(len(gamma_mag)):
                print(f"{i+1:4} | {gamma_mag[i]:.6f}             | {beta_mag[i]:.6f}")

    if not found_film:
        # Check standard attention as fallback
        if "attention_map" in data:
            attn = np.mean(data["attention_map"], axis=0)
            feat_names = data["feature_names"] if "feature_names" in data else None
            print(f"\n✨ Attention Map Detected (Shape: {attn.shape})")
            print("To view attention numericals, use: python inspect_internals.py")

def main():
    files = glob.glob("result*/**/*internals.npz", recursive=True)
    if not files:
        print("No .npz files found.")
        return
        
    for i, f in enumerate(files):
        print(f"{i+1}. {f}")
        
    choice = input("\nSelect file index: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            print_film_stats(files[idx])
    except:
        print("Invalid input.")

if __name__ == "__main__":
    main()
