
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

class Evaluator:
    def __init__(self, result_dir: Path, fig_dir: Path):
        self.result_dir = result_dir
        self.fig_dir = fig_dir
        for d in [self.result_dir, self.fig_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates RMSE and MAPE."""
        # Flat metrics
        yt_flat = y_true.reshape(-1)
        yp_flat = y_pred.reshape(-1)
        
        rmse = float(np.sqrt(mean_squared_error(yt_flat, yp_flat)))
        mape = float(mean_absolute_percentage_error(yt_flat, yp_flat) * 100.0)
        
        return {"rmse": rmse, "mape": mape}

    def save_prediction_csv(self, 
                          run_name: str, 
                          preds: np.ndarray, 
                          trues: np.ndarray, 
                          info: Dict[str, Any], 
                          config: Dict[str, Any]) -> Path:
        """
        Saves predictions to CSV with full timestamp and metadata.
        Reconstructs absolute time using info from data_loader.
        """
        out_path = self.result_dir / f"{run_name}.csv"
        
        test_starts = info["test_starts"]
        all_ts = info["all_timestamp"]
        all_time = info["all_time"]
        seq_len = config["seq_len"]
        horizon = config["pred_len"]
        
        rows = []
        n_samples = len(preds)
        
        for s in range(n_samples):
            start_idx = int(test_starts[s])
            
            # Slice from original full series
            # Target window is [start_idx + seq_len : start_idx + seq_len + horizon]
            t_start = start_idx + seq_len
            t_end   = t_start + horizon
            
            ts_slice   = all_ts.iloc[t_start : t_end].to_numpy()
            time_slice = all_time.iloc[t_start : t_end].to_numpy()
            
            for h in range(horizon):
                ts_val = pd.to_datetime(ts_slice[h])
                rows.append([
                    ts_val,
                    ts_val.date().isoformat(),
                    int(time_slice[h]),
                    int(ts_val.hour),
                    int(ts_val.minute),
                    int(h + 1),
                    float(trues[s, h]),
                    float(preds[s, h])
                ])

        df_out = pd.DataFrame(rows, columns=[
            "timestamp", "date", "time", "hour", "minute", "horizon", "y_true", "y_pred"
        ])
        
        # Add run metadata columns
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool)):
                df_out[k] = v
                
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"📦 Predictions saved: {out_path}")
        return out_path

    def plot_results(self, run_name: str, pred_csv_path: Path):
        """Plots aggregated Time vs True/Pred."""
        df = pd.read_csv(pred_csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Aggregate if there are overlaps (e.g. sliding window) or just to sort
        df_plot = df.groupby("timestamp", as_index=False)[["y_true", "y_pred"]].mean().sort_values("timestamp")
        
        fig_path = self.fig_dir / f"{run_name}.png"
        
        plt.figure(figsize=(14, 6))
        plt.plot(df_plot["timestamp"], df_plot["y_true"], label="True", alpha=0.7)
        plt.plot(df_plot["timestamp"], df_plot["y_pred"], label="Pred", alpha=0.7, linestyle="--")
        plt.title(f"Forecast Result: {run_name}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"🖼️ Figure saved: {fig_path}")

    def append_summary(self, summary_path: Path, row: Dict[str, Any]):
        """Appends a row to the master summary CSV."""
        df_row = pd.DataFrame([row])
        
        if summary_path.exists():
            old = pd.read_csv(summary_path)
            # Ensure columns match
            new_cols = [c for c in df_row.columns if c not in old.columns]
            for c in new_cols:
                old[c] = np.nan
            
            # Align new row to old columns + new columns
            combined = pd.concat([old, df_row], ignore_index=True)
        else:
            combined = df_row
            
        combined.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"📝 Summary updated: {summary_path}")
