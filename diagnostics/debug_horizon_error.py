import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def evaluate_horizon_difficulty(result_csv_path):
    print(f"Loading results from: {result_csv_path}")
    if not os.path.exists(result_csv_path):
        print(f"❌ File not found: {result_csv_path}")
        return

    df = pd.read_csv(result_csv_path)
    
    # Check required columns
    required_cols = ["horizon", "y_true", "y_pred"]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Missing columns! Found: {df.columns}. Expected: {required_cols}")
        return
    
    print("✅ Columns found. Calculating error by horizon step...")
    
    steps = sorted(df["horizon"].unique())
    rmses = []
    mapes = []
    
    print(f"Found horizon steps: {steps}")
    
    for h in steps:
        step_df = df[df["horizon"] == h]
        y_true = step_df["y_true"].values
        y_pred = step_df["y_pred"].values
        
        if len(y_true) == 0:
            continue
            
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        rmses.append(rmse)
        mapes.append(mape)
        # Optional: Print detail
        # print(f"Step {h}: RMSE={rmse:.4f}, MAPE={mape:.4f}%")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Horizon Step (Hours Ahead)')
    ax1.set_ylabel('RMSE', color='tab:red')
    ax1.plot(steps, rmses, color='tab:red', marker='o', label='RMSE', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx() 
    ax2.set_ylabel('MAPE (%)', color='tab:blue')
    ax2.plot(steps, mapes, color='tab:blue', marker='s', label='MAPE', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(f'Prediction Difficulty by Horizon\n({os.path.basename(result_csv_path)})')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    fig.tight_layout()
    
    save_path = result_csv_path.replace(".csv", "_error_by_step.png")
    plt.savefig(save_path, dpi=150)
    print(f"✅ Plot saved to: {save_path}")
    # plt.show() # Non-blocking

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        evaluate_horizon_difficulty(path)
    else:
        # Default auto-discovery
        base_dir = r"C:\Users\2213144\practice\result_csv"
        # Find most recent file if possible, or specific default
        last_run = r"C:\Users\2213144\practice\result_csv\Hybrid_FiLM_Ablation\Hybrid_FiLM_Ablation_all_separated.csv"
        evaluate_horizon_difficulty(last_run)
