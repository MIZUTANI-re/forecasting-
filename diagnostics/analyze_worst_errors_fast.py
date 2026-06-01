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
import sys

# Define models list
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
    "MLP", 
    "grid_tst",
    "MMK"
]

BASE_DIR = Path(r"c:\Users\2213144\practice\result_csv")
LOG_FILE = Path(r"c:\Users\2213144\practice\analysis_debug.log")

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
    print(msg)

def analyze_model_robust(model_name):
    log(f"--- Analyzing {model_name} ---")
    folder_name = model_name
    if model_name == "MLP":
        folder_name = "transformer_oneshot"
    # TransformerFixed is TransformerFixed in dir, so no change needed
    
    model_dir = BASE_DIR / folder_name
    if not model_dir.exists():
        log(f"Dir not found: {model_dir}")
        return None

    # Priority: unified -> all -> spring
    files = list(model_dir.glob("*_all_unified.csv"))
    if not files:
        files = list(model_dir.glob("*_spring_unified.csv"))
    if not files:
        files = list(model_dir.glob("*_all_*.csv"))
    if not files:
        files = list(model_dir.glob("*_spring_*.csv"))
    
    if not files:
        log(f"No CSV files found in {model_dir}")
        return None
        
    target_file = files[0]
    log(f"Processing {target_file}")
    
    try:
        df = pd.read_csv(target_file, low_memory=False)
        log(f"Columns: {list(df.columns)}")
        
        # Check for Long Format (y_true, y_pred) - PRIORITIZE THIS
        if "y_true" in df.columns and "y_pred" in df.columns:
            log("Detected Long Format (y_true/y_pred)")
            
            pred_len = 24
            if "pred_len" in df.columns:
                try: pred_len = int(df["pred_len"].iloc[0])
                except: pass
            elif "horizon" in df.columns:
                 try: pred_len = int(df["horizon"].max())
                 except: pass
            
            log(f"Using pred_len={pred_len}")
            
            y_pred = pd.to_numeric(df["y_pred"], errors='coerce')
            y_true = pd.to_numeric(df["y_true"], errors='coerce')
            
            sq_err = (y_pred - y_true)**2
            
            # log(f"Sq Err NaNs: {sq_err.isna().sum()} / {len(sq_err)}")
            
            rolling_mse = sq_err.rolling(window=pred_len, min_periods=int(pred_len/2)).mean()
            
            if rolling_mse.isna().all():
                log("All rolling MSE are NaN")
                return None
                
            rolling_rmse = np.sqrt(rolling_mse)
            
            worst_idx = rolling_mse.idxmax()
            
            if pd.isna(worst_idx):
                log("worst_idx is NaN")
                return None
            
            worst_idx = int(worst_idx)
            worst_rmse = rolling_rmse.iloc[worst_idx] 
            
            log(f"Worst Idx: {worst_idx}, Worst RMSE: {worst_rmse}")
            
            start_idx = max(0, worst_idx - pred_len + 1)
            
            date_val = "Unknown"
            if "date" in df.columns:
                date_val = df.iloc[start_idx]["date"]
            elif "timestamp" in df.columns:
                date_val = df.iloc[start_idx]["timestamp"]
            
            window_slice = df.iloc[start_idx : worst_idx+1]
            w_true = pd.to_numeric(window_slice["y_true"], errors='coerce')
            w_pred = pd.to_numeric(window_slice["y_pred"], errors='coerce')
            
            t_mean = w_true.mean()
            p_mean = w_pred.mean()
            
            return {
                "model": model_name,
                "date": str(date_val),
                "worst_rmse": worst_rmse,
                "true_mean": t_mean,
                "pred_mean": p_mean,
                "diff_mean": p_mean - t_mean
            }

        # Check for Wide Format (pred_0, pred_1...)
        pred_cols = [c for c in df.columns if c.startswith("pred_") and c != "pred_len"]
        true_cols = [c for c in df.columns if c.startswith("true_")]
        
        if pred_cols:
            log("Detected Wide Format")
            # WIDE FORMAT LOGIC
            if not true_cols:
                if "target" in df.columns:
                    true_cols = [c for c in df.columns if c.startswith("true_")]
            
            if len(pred_cols) > 0:
                if not true_cols or len(true_cols) != len(pred_cols):
                     log(f"Mismatch: pred_cols={len(pred_cols)}, true_cols={len(true_cols)}")
                     return None

                preds = df[pred_cols].apply(pd.to_numeric, errors='coerce').values
                trues = df[true_cols].apply(pd.to_numeric, errors='coerce').values
                
                mse = np.nanmean((preds - trues)**2, axis=1)
                rmse = np.sqrt(mse)
                
                if np.all(np.isnan(rmse)):
                    log("All RMSE are NaN")
                    return None
                    
                worst_idx = np.nanargmax(rmse)
                worst_rmse = rmse[worst_idx]
                
                date_val = "Unknown"
                if "date" in df.columns: date_val = df.iloc[worst_idx]["date"]
                elif "timestamp" in df.columns: date_val = df.iloc[worst_idx]["timestamp"]
                
                t_mean = np.nanmean(trues[worst_idx])
                p_mean = np.nanmean(preds[worst_idx])
                
                return {
                    "model": model_name,
                    "date": str(date_val),
                    "worst_rmse": worst_rmse,
                    "true_mean": t_mean,
                    "pred_mean": p_mean,
                    "diff_mean": p_mean - t_mean
                }
            
        else:
            log("No valid columns found")
            return None

    except Exception as e:
        log(f"EXCEPTION: {e}")
        return None

def main():
    if LOG_FILE.exists():
        os.remove(LOG_FILE)
        
    results = []
    print(f"{'Model':<30} | {'Date':<20} | {'Max RMSE':<10} | {'True':<10} | {'Pred':<10} | {'Diff':<10}")
    print("-" * 100)
    
    for model in FULL_MODEL_LIST:
        res = analyze_model_robust(model)
        if res:
            results.append(res)
            print(f"{res['model']:<30} | {res['date']:<20} | {res['worst_rmse']:<10.2f} | {res['true_mean']:<10.1f} | {res['pred_mean']:<10.1f} | {res['diff_mean']:<10.1f}")
        else:
            print(f"{model:<30} | {'N/A':<20} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
    
    md_output = "# Model Worst-Case Analysis\n\n"
    md_output += "> **Note:** Analysis based on sliding window of prediction length (24) across aggregated results.\n\n"
    md_output += "| Model | Worst Window Start | Max Window RMSE | True (Mean) | Pred (Mean) | Diff |\n"
    md_output += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for res in results:
        md_output += f"| **{res['model']}** | {res['date']} | **{res['worst_rmse']:.2f}** | {res['true_mean']:.1f} | {res['pred_mean']:.1f} | {res['diff_mean']:+.1f} |\n"
        
    with open(r"c:\Users\2213144\practice\worst_case_analysis.md", "w", encoding="utf-8") as f:
        f.write(md_output)

if __name__ == "__main__":
    main()
