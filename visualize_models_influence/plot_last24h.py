import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from visualize_models_influence folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
import re

# ============================================================
# Configuration
# ============================================================
SEASONS_TO_PROCESS = ["all", "spring", "summer", "autumn", "winter"]
# Default Paths (will be updated dynamically)
BASE_DIR = Path(r"C:\Users\2213144\practice")
RESULT_DIR_POWER = BASE_DIR / "result_csv"
RESULT_DIR_RESIDUAL = BASE_DIR / "result_csv_residual"

OUTPUT_DIR_POWER = BASE_DIR / "last_24"
OUTPUT_DIR_RESIDUAL = BASE_DIR / "last_24_residual"


# Data aggregation list
aggregated_dfs = []

# ============================================================

def get_available_models(result_dir):
    # List subdirectories in result_dir
    return sorted([d.name for d in result_dir.iterdir() if d.is_dir()])

def select_models(available_models):
    print("\n📊 Available Models:")
    for i, m in enumerate(available_models):
        print(f"  {i+1}. {m}")
    
    print("\nWhich models do you want to include?")
    print("  - Enter numbers separated by commas or spaces (e.g., '1 3 5')")
    print("  - Enter 'all' for all models")
    
    choice = input("Select models > ").strip().lower()
    
    if choice == 'all':
        return available_models
    
    selected = []
    try:
        # Split by comma or whitespace, filter empty strings
        tokens = [t for t in re.split(r'[,\s]+', choice) if t]
        indices = [int(x) - 1 for x in tokens]
        
        for idx in indices:
            if 0 <= idx < len(available_models):
                selected.append(available_models[idx])
            else:
                print(f"⚠ Skipping invalid index: {idx+1}")
    except ValueError:
        print("❌ Invalid input format. Using all models as fallback.")
        return available_models
        
    if not selected:
        print("⚠ No valid models selected. Using all models.")
        return available_models
        
    return selected

def plot_last_24h():
    print("="*60)
    print("   📊 Plot Results (Last 24H)   ")
    print("="*60)
    
    print("\nSelect Result Type:")
    print("  1. Power Demand (Default)")
    print("  2. Residual Demand")
    r_choice = input("Enter number (1-2): ").strip()
    
    if r_choice == '2':
        target_name = "Residual Demand"
        RESULT_DIR = RESULT_DIR_RESIDUAL
        OUTPUT_DIR = OUTPUT_DIR_RESIDUAL
    else:
        target_name = "Power Demand"
        RESULT_DIR = RESULT_DIR_POWER
        OUTPUT_DIR = OUTPUT_DIR_POWER
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Target: {target_name}")
    print(f"📂 Reading from: {RESULT_DIR}")
    print(f"📂 Output to:   {OUTPUT_DIR}")
    
    if not RESULT_DIR.exists():
        print(f"❌ Result directory not found: {RESULT_DIR}")
        return

    available_models = get_available_models(RESULT_DIR)
    if not available_models:
        print("❌ No model directories found in result directory!")
        return

    models_to_compare = select_models(available_models)
    print(f"\n✅ Selected Models ({len(models_to_compare)}): {models_to_compare}")

    for SEASON in SEASONS_TO_PROCESS:
        print(f"\n🎨 Processing Season: [{SEASON}]")
        
        plt.figure(figsize=(15, 6))
        
        # Track if we have plotted ground truth
        ground_truth_plotted = False
        found_any = False

        for model_name in models_to_compare:
            # Construct path: result_csv/ModelName/*_{SEASON}_*.csv
            # We use glob to be flexible with filenames
            
            # Pattern: any file containing the season name inside the model's folder
            model_dir = RESULT_DIR / model_name
            search_pattern = str(model_dir / f"*_{SEASON}_*.csv")
            
            files = glob.glob(search_pattern)
            
            if not files:
                # Debug info only if strictly necessary, to keep output clean?
                # But user reported missing models, so let's show what's happening.
                # print(f"  ⚠ Skipped {model_name}: No CSV found for {SEASON}")
                continue
            
            # Prefer "unified" file if multiple exist
            target_file = files[0]
            for f in files:
                if "unified" in f:
                    target_file = f
                    break
            
            print(f"  - Plotting {model_name}...")
            
            try:
                df = pd.read_csv(target_file)
                
                # Ensure sorting? NO, this mixes overlapping windows.
                # The CSV is ordered by [Sample1_H1..24, Sample2_H1..24...].
                # We want the LAST sample's 24 horizons.
                # if "timestamp" in df.columns:
                #    df["timestamp"] = pd.to_datetime(df["timestamp"])
                #    # df = df.sort_values(["timestamp", "horizon"]) # <--- CAUSES DUPLICATES
                
                # Just take the tail of the naturally ordered file
                last_24 = df.tail(24).copy()
                
                # Add metadata for aggregation
                last_24["Season"] = SEASON
                last_24["Model"] = model_name
                aggregated_dfs.append(last_24)
                
                # Plot Ground Truth only once
                # Use zorder=10 to keep it on top, color=black for visibility
                if not ground_truth_plotted:
                    plt.plot(range(24), last_24["y_true"].values, label="Actual (Ground Truth)", color="black", linewidth=2.5, linestyle="-", zorder=10)
                    ground_truth_plotted = True
                
                # Plot Prediction
                plt.plot(range(24), last_24["y_pred"].values, label=model_name, linewidth=1.5, alpha=0.8)
                found_any = True
                
            except Exception as e:
                print(f"❌ Error reading {model_name}: {e}")

        if found_any:
            plt.title(f"Forecast Comparison (Last 24H) - [{SEASON}]", fontsize=16)
            plt.xlabel("Horizon Step (Hour)", fontsize=12)
            plt.ylabel(target_name, fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = OUTPUT_DIR / f"last24_plot_{SEASON}.png"
            plt.savefig(save_path, dpi=300)
            print(f"  ✅ Saved Plot: {save_path}")
            plt.close()
        else:
            print(f"  ❌ No valid data found for season '{SEASON}'. Skipping plot.")
            plt.close()

    
    # Save aggregated CSV
    if aggregated_dfs:
        final_df = pd.concat(aggregated_dfs, ignore_index=True)
        # Reorder columns
        cols = ["Season", "Model", "timestamp", "horizon", "y_true", "y_pred"]
        existing_cols = [c for c in cols if c in final_df.columns]
        other_cols = [c for c in final_df.columns if c not in existing_cols]
        final_df = final_df[existing_cols + other_cols]
        
        output_csv = OUTPUT_DIR / "last_24.csv"
        final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n📦 Aggregated CSV saved to: {output_csv}")
    else:
        print("\n⚠ No data was aggregated (check if models ran successfully).")

if __name__ == "__main__":
    plot_last_24h()
