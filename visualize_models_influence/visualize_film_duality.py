import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from visualize_models_influence folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import glob

import pandas as pd

# Style
sns.set_theme(style="whitegrid")
OUTPUT_DIR = "analysis_film_duality"
os.makedirs(OUTPUT_DIR, exist_ok=True)

summary_data = []

def analyze_duality(npz_path):
    data = np.load(npz_path)
    filename = Path(npz_path).name
    parts = filename.replace("_internals.npz", "").split("_")
    model_name = "_".join(parts[:-2])
    season = parts[-2]
    
    print(f"🧐 Analyzing FiLM Duality: {model_name} ({season})")
    
    # 1. Extract Gamma and convert to Scale Factor (Mapping to 0.0 ~ 1.0 range based on logic)
    # scale = 1 + tanh(gamma) -> range (0, 2)
    # We normalize to (0, 1) where 0.5 is 'Stable/Normal'
    
    results = {}
    
    for group in ["tv", "vt"]:
        key = f"gamma_{group}"
        if key in data:
            gammas = data[key] # (Batch, Horizon, D)
            # Map to 0-1 range for scholarly explanation
            # 0.5 * (1 + tanh(gamma))
            scales = 0.5 * (1 + np.tanh(gammas))
            
            # Aggregate: Mean intensity per sample
            sample_intensity = np.mean(scales, axis=(1, 2))
            results[group] = sample_intensity
            
            # Statistics
            mean_val = np.mean(sample_intensity)
            variance = np.var(sample_intensity)
            std_val = np.std(sample_intensity)
            min_val = np.min(sample_intensity)
            max_val = np.max(sample_intensity)
            
            print(f"   -> {group.upper()} Mean Intensity: {mean_val:.4f}, Variance: {variance:.6f}")
            
            summary_data.append({
                "季節": season,
                "FiLMモード": group.upper(),
                "介入レベル平均 (Mean)": round(mean_val, 4),
                "分散 (Variance)": round(variance, 6),
                "標準偏差 (Std)": round(std_val, 4),
                "最小値 (Min)": round(min_val, 4),
                "最大値 (Max)": round(max_val, 4)
            })

    if not results:
        print("   ⚠ No FiLM Gamma values found in this file.")
        return

    # 2. Plotting the Duality Distribution
    plt.figure(figsize=(12, 7))
    
    plot_data = []
    labels = []
    for g, vals in results.items():
        plot_data.extend(vals)
        labels.extend([f"FiLM {g.upper()} (Scale)"] * len(vals))
        
    sns.violinplot(x=labels, y=plot_data, inner="quart", palette="muted")
    plt.axhline(0.5, color='gray', linestyle='--', label="Stable (Normal)")
    plt.axhline(0.2, color='red', linestyle=':', label="Blocking Zone")
    plt.axhline(0.8, color='green', linestyle=':', label="Amplifying Zone")
    
    plt.title(f"FiLM Duality Analysis: {season.capitalize()} Season\n(Mechanism: How Model Uses Time vs. Variables)", fontsize=14)
    plt.ylabel("Intervention Level (0: Block, 0.5: Stable, 1: Amplify)")
    plt.ylim(-0.1, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.text(-0.4, 0.05, "Block Noise", color="red", fontweight="bold")
    plt.text(-0.4, 0.52, "Stable Coop", color="gray", fontweight="bold")
    plt.text(-0.4, 0.92, "Signal Amplify", color="green", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_{season}_duality_violin.png")
    plt.close()
    
    # --- New: 24-Hour Average Profile (Horizon Steps) ---
    plt.figure(figsize=(12, 6))
    for group in ["tv", "vt"]:
        if group in results:
            key = f"gamma_{group}"
            # results[group] here is sample_intensity (averaged over steps already)
            # We need the per-step average
            scales = 0.5 * (1 + np.tanh(data[key])) # (Batch, Horizon, D)
            horizon_profile = np.mean(scales, axis=(0, 2)) # Average over Batch and D
            plt.plot(np.arange(1, len(horizon_profile)+1), horizon_profile, marker='o', label=f"FiLM {group.upper()} Average")
            
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.title(f"Model Strategy Profile (24-Hour Horizon Average): {season.capitalize()}")
    plt.xlabel("Hours Ahead (Prediction Horizon)")
    plt.ylabel("Intervention Level (0: Block, 0.5: Stable, 1: Amplify)")
    plt.ylim(0.3, 0.7) # Focus on the active range
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_{season}_24h_profile.png")
    plt.close()
    print(f"   -> Saved 24-hour average profile plot")

    # 3. Switching Behavior Plot (First 100 samples)
    if "vt" in results:
        plt.figure(figsize=(15, 5))
        plt.plot(results["vt"][:200], label="V2T (Variable influencing Time)", color="purple", alpha=0.8)
        if "tv" in results:
            plt.plot(results["tv"][:200], label="T2V (Time influencing Variable)", color="orange", alpha=0.6)
            
        plt.title(f"Dynamic Switching Behavior (Sample Sequence: First 200) - {season.capitalize()}")
        plt.xlabel("Sample Index (Chronological)")
        plt.ylabel("Intervention Level")
        plt.axhline(0.5, color='black', alpha=0.2)
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/{model_name}_{season}_switching.png")
        plt.close()
        print(f"   -> Saved duality and switching plots to {OUTPUT_DIR}")

def main():
    files = glob.glob("result*/**/*internals.npz", recursive=True)
    if not files:
        print("❌ No data files found. Please run main.py first.")
        return
        
    for f in files:
        analyze_duality(f)

    # --- Save Summary CSV ---
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        csv_path = f"{OUTPUT_DIR}/film_duality_summary.csv"
        df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ Summary CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
