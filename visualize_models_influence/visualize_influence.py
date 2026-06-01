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

# Style settings
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

OUTPUT_DIR = "analysis_influence_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_influence(npz_path):
    data = np.load(npz_path)
    filename = Path(npz_path).name
    parts = filename.replace("_internals.npz", "").split("_")
    model_name = "_".join(parts[:-2])
    season = parts[-2]
    
    print(f"📊 Analyzing Influence for: {model_name} ({season})")
    
    feature_names = data["feature_names"] if "feature_names" in data else None
    
    # --- 1. Past Demand vs. Features (Gating Balance) ---
    for alpha_key in ["alpha", "gate_alpha"]:
        if alpha_key in data:
            alpha = np.mean(data[alpha_key], axis=0).flatten()
            steps = np.arange(1, len(alpha) + 1)
            
            plt.figure(figsize=(10, 5))
            plt.bar(steps, alpha, label="Past Demand (Temporal Branch)", color='skyblue', alpha=0.7)
            plt.bar(steps, 1-alpha, bottom=alpha, label="Exogenous Features (Variable Branch)", color='salmon', alpha=0.7)
            
            plt.title(f"Contribution Balance: Past Demand vs. Features ({model_name})")
            plt.xlabel("Hours Ahead (Horizon)")
            plt.ylabel("Influence Ratio")
            plt.ylim(0, 1.1)
            plt.legend(loc='upper right')
            plt.savefig(f"{OUTPUT_DIR}/{model_name}_{season}_balance.png")
            plt.close()
            print(f"   -> Saved Influence Balance Plot")

    # --- 2. Specific Feature Importance (Variable Attention) ---
    if "attention_map" in data:
        # attention_map is (Variable, Variable)
        attn = np.mean(data["attention_map"], axis=0)
        
        # We want to know how much each feature (source) influences the target (power_demand)
        # In iTransformer, target is usually the first variable or the one specified by target_idx
        # Let's find 'power_demand' or 'residual_demand' in feature_names
        target_idx = -1
        if feature_names is not None:
            names = [n.lower() for n in feature_names]
            for i, n in enumerate(names):
                if 'demand' in n:
                    target_idx = i
                    break
        
        # If target found, extract its attention row (who influences target)
        if target_idx != -1:
            influence_on_target = attn[target_idx] # Row target_idx: Source influence on this target
            
            # Sort importance
            indices = np.argsort(influence_on_target)[::-1]
            sorted_names = [feature_names[i] for i in indices] if feature_names is not None else [f"Var_{i}" for i in indices]
            sorted_vals = influence_on_target[indices]
            
            plt.figure(figsize=(10, 8))
            colors = ['red' if 'demand' in n.lower() else 'steelblue' for n in sorted_names]
            sns.barplot(x=sorted_vals, y=sorted_names, palette=colors)
            plt.axvline(x=1.0/len(sorted_vals), color='gray', linestyle='--', label="Uniform Attention")
            plt.title(f"Feature Influence Ranking on Prediction ({model_name})")
            plt.xlabel("Attention weight (Influence Degree)")
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/{model_name}_{season}_feature_importance.png")
            plt.close()
            print(f"   -> Saved Feature Importance Ranking")
            
            # --- Print text summary for the user ---
            print("\n✨ Feature Influence Ranking (Top 10):")
            print("-" * 40)
            for i in range(min(10, len(sorted_names))):
                print(f"{i+1:2}. {sorted_names[i]:25} | {sorted_vals[i]:.2%}")
            print("-" * 40)

    # --- 3. FiLM Sensitivity (Feature Impact Strength) ---
    if "gamma_tv" in data or "gamma_vt" in data:
        # FiLM represents how much features modulate the demand representation
        # Magnitude of gamma indicates sensitivity to features
        g_key = "gamma_tv" if "gamma_tv" in data else "gamma_vt"
        gamma = np.mean(np.abs(data[g_key]), axis=0) # (Horizon, D)
        impact_strength = np.mean(gamma, axis=1) # Mean over d_model
        
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(1, len(impact_strength)+1), impact_strength, marker='s', color='darkred')
        plt.title(f"Feature Influence Strength (FiLM Sensitivity) over Horizon")
        plt.xlabel("Hours Ahead")
        plt.ylabel("Sensitivity Magnitude")
        plt.grid(True)
        plt.savefig(f"{OUTPUT_DIR}/{model_name}_{season}_film_sensitivity.png")
        plt.close()
        print(f"   -> Saved FiLM Sensitivity Plot")

def main():
    files = glob.glob("result*/**/*internals.npz", recursive=True)
    if not files:
        print("❌ No internal state files found.")
        return
        
    for f in files:
        analyze_influence(f)
    print(f"\n✅ Influence Analysis Complete. Results saved in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
