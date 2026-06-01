import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Define models and their folder names
# Note: "MLP" was renamed from "transformer_oneshot" in code, but folder might still be "transformer_oneshot"
# Based on list_dir output:
FULL_MODEL_LIST = [
    "FiLM",
    "iTransformer",
    "TransformerFixed",
    "Hybrid_FiLM_Ablation",
    "Hybrid_Predict_Fusion",
    "Hybrid_NoHQ_Predict_Fusion",
    "Hybrid_Gated_FeatureFusion",
    "Hybrid_Gated_QueryFusion",
    "Hybrid_KAN_Gated_FeatureFusion",
    "htmformer",
    "MLP", # We will check "transformer_oneshot" folder for this
    "grid_tst",
    "MMK"
]

BASE_DIR = Path(r"c:\Users\2213144\practice\result_csv")

def calculate_row_rmse(row, pred_cols, true_cols):
    preds = row[pred_cols].values.astype(float)
    trues = row[true_cols].values.astype(float)
    mse = np.mean((preds - trues) ** 2)
    return np.sqrt(mse)

def analyze_model(model_name):
    # Handle folder name mapping
    folder_name = model_name
    if model_name == "MLP":
        folder_name = "transformer_oneshot"
    
    model_dir = BASE_DIR / folder_name
    if not model_dir.exists():
        print(f"❌ Directory not found: {model_dir}")
        return None

    # Find the target CSV file: Look for "*_all_*.csv" first (Standardized run)
    # File pattern: {MODEL_NAME}_{SEASON}_{dataset_mode}.csv
    # e.g., FiLM_all_unified.csv
    
    files = list(model_dir.glob("*_all_*.csv"))
    if not files:
        # Fallback to spring if all not found
        files = list(model_dir.glob("*_spring_*.csv"))
    
    if not files:
        print(f"⚠ No prediction CSV found for {model_name} (checked 'all' and 'spring')")
        return None
        
    # Pick the most recent file if multiple
    # But usually there's only one relevant one per season/mode combination
    # We prefer "unified" mode for most, but MMK uses "separated"
    target_file = files[0] # Simplification
    # print(f"📂 Processing {model_name} from {target_file.name}")
    
    try:
        df = pd.read_csv(target_file)
        
        # Identify columns
        # Prediction cols start with "pred_", True cols start with "true_"
        pred_cols = [c for c in df.columns if c.startswith("pred_")]
        true_cols = [c for c in df.columns if c.startswith("true_")]
        
        if not pred_cols:
            print(f"⚠ No pred columns in {target_file.name}")
            return None

        # Calculate RMSE for each row (window)
        # Using list comprehension for speed
        rmses = []
        for idx, row in df.iterrows():
            rmse = calculate_row_rmse(row, pred_cols, true_cols)
            rmses.append(rmse)
            
        df["window_rmse"] = rmses
        
        # Find worst row
        worst_idx = df["window_rmse"].idxmax()
        worst_row = df.loc[worst_idx]
        
        return {
            "model": model_name,
            "filename": target_file.name,
            "date": worst_row.get("date", "Unknown"), # Assuming 'date' column exists, usually index or first col
            "worst_rmse": worst_row["window_rmse"],
            "true_mean": np.mean(worst_row[true_cols].values.astype(float)),
            "pred_mean": np.mean(worst_row[pred_cols].values.astype(float)),
            "diff_mean": np.mean(worst_row[pred_cols].values.astype(float) - worst_row[true_cols].values.astype(float))
        }

    except Exception as e:
        print(f"❌ Error analyzing {model_name}: {e}")
        return None

def main():
    results = []
    print(f"{'Model':<30} | {'Date':<20} | {'Worst RMSE':<10} | {'True (Mean)':<10} | {'Pred (Mean)':<10} | {'Diff':<10}")
    print("-" * 100)
    
    for model in FULL_MODEL_LIST:
        res = analyze_model(model)
        if res:
            results.append(res)
            print(f"{res['model']:<30} | {res['date']:<20} | {res['worst_rmse']:<10.4f} | {res['true_mean']:<10.2f} | {res['pred_mean']:<10.2f} | {res['diff_mean']:<10.2f}")
    
    # Save to Markdown for the User
    md_output = "# Worst Prediction Cases by Model\n\n"
    md_output += "| Model | Worst Date | Window RMSE | True (Mean) | Pred (Mean) | Diff (Pred-True) |\n"
    md_output += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for res in results:
        diff_str = f"{res['diff_mean']:.2f}"
        if res['diff_mean'] > 0:
            diff_str = f"+{diff_str}"
            
        md_output += f"| **{res['model']}** | {res['date']} | **{res['worst_rmse']:.2f}** | {res['true_mean']:.1f} | {res['pred_mean']:.1f} | {diff_str} |\n"
        
    with open(r"c:\Users\2213144\practice\worst_case_analysis.md", "w", encoding="utf-8") as f:
        f.write(md_output)
    
    print("\n✅ Analysis complete. Saved to worst_case_analysis.md")

if __name__ == "__main__":
    main()
