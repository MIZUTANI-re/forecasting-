import os
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# Settings & Paths
# ============================================================
BASE_DIR = Path(r"c:\Users\2213144\practice")
RESULT_DIR = BASE_DIR / "result_csv"
THESIS_DIR = BASE_DIR / "thesis"
THESIS_DIR.mkdir(parents=True, exist_ok=True)

# 比較対象モデル
MODELS = ["TransformerFixed", "iTransformer", "Hybrid_Predict_Fusion"]

MODEL_DISPLAY_NAMES = {
    "TransformerFixed": "Transformer",
    "iTransformer": "iTransformer",
    "Hybrid_Predict_Fusion": "Hybrid",
}

# 大予測ミス（大外し）とみなすエラーのしきい値 (万kW単位)
# 需要の約10〜15%程度である 30万kW を標準とする
LARGE_ERROR_THRESHOLD = 30.0

def read_csv_auto(path) -> pd.DataFrame:
    path = str(path)
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    return pd.read_csv(path)

def analyze_worst_days():
    print("=" * 70)
    print("   [Analysis] Worst Forecast Days and Model Robustness (GW & Holidays)   ")
    print("=" * 70)
    
    model_dfs = {}
    
    # 1. 各モデルの予測結果ファイルを読み込み
    for model in MODELS:
        model_dir = RESULT_DIR / model
        if not model_dir.exists():
            continue
            
        # allシーズンの結果を探す
        files = list(model_dir.glob("*_all_unified.csv"))
        if not files:
            files = list(model_dir.glob("*_all_*.csv"))
        if not files:
            files = list(model_dir.glob("*.csv"))
            
        if files:
            # 最新のファイルをロード
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            target_file = files[0]
            print(f"Loaded {model:<22} from: {target_file.name}")
            
            df = read_csv_auto(target_file)
            
            # 時間列の自動検出
            time_col = None
            for col in ["timestamp", "datetime", "date", "target_datetime"]:
                if col in df.columns:
                    time_col = col
                    break
                    
            if time_col is None:
                print(f"  [Warning] Missing timestamp in {model}")
                continue
                
            df[time_col] = pd.to_datetime(df[time_col].astype(str).str.strip(), errors="coerce")
            df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
            df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
            df = df.dropna(subset=[time_col, "y_true", "y_pred"]).copy()
            
            # 【重要】予測のスライディングウィンドウによる重複日時を、平均化して一意（1日時＝1レコード）にします
            df_unique = df.groupby(time_col, as_index=False).agg({
                "y_true": "mean",
                "y_pred": "mean"
            }).copy()
            
            # 誤差計算
            df_unique["date_only"] = df_unique[time_col].dt.date
            df_unique["error"] = df_unique["y_pred"] - df_unique["y_true"]
            df_unique["abs_error"] = df_unique["error"].abs()
            df_unique["sq_error"] = df_unique["error"]**2
            
            model_dfs[model] = df_unique

    if not model_dfs:
        print("No prediction data loaded. Exiting.")
        return

    # 2. 日付ごとにグループ化してエラー統計を算出
    daily_summaries = {}
    for model, df in model_dfs.items():
        # 日ごとのRMSEと最大絶対誤差を計算
        daily = df.groupby("date_only").agg(
            Actual_Mean=("y_true", "mean"),
            Pred_Mean=("y_pred", "mean"),
            Max_Absolute_Error=("abs_error", "max"),
            Daily_RMSE=("sq_error", lambda x: np.sqrt(np.mean(x))),
            Count=("y_true", "count")
        ).copy()
        
        # 1日のデータ数が24点揃っている日のみ（端数日・欠損日を除く）
        daily = daily[daily["Count"] == 24].copy()
        daily_summaries[model] = daily

    # 3. 堅牢性評価（大予測エラーの発生頻度カウント）
    print(f"\n=== [Robustness Evaluation] Large Forecast Error Days (Threshold: >= {LARGE_ERROR_THRESHOLD} 10^4 kW) ===")
    print("-" * 90)
    print(f"{'Model Name':<25} | {'Total Test Days':<15} | {'Large Error Days':<18} | {'Error Day Rate (%)':<20}")
    print("-" * 90)
    
    for model in MODELS:
        if model not in daily_summaries:
            continue
        daily = daily_summaries[model]
        total_days = len(daily)
        large_error_days = len(daily[daily["Max_Absolute_Error"] >= LARGE_ERROR_THRESHOLD])
        rate = (large_error_days / total_days) * 100.0
        
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        print(f"{display_name:<25} | {total_days:<15} | {large_error_days:<18} | {rate:<20.2f}%")
    print("-" * 90)
    print("  * Note: 'Large Error Day' represents a day where the prediction error exceeded the threshold in at least 1 hour.")

    # 4. 各モデルのワースト5日の特定（最も大きく予測を外した日付）
    print(f"\n=== [Worst Days] Top 5 Worst Forecast Days for Each Model ===")
    for model in MODELS:
        if model not in daily_summaries:
            continue
        daily = daily_summaries[model]
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        print(f"\n* [Model: {display_name}]")
        print(f"  {'Rank':<5} | {'Date (Day)':<16} | {'Daily RMSE':<12} | {'Max Absolute Error':<20} | {'Actual Mean':<12} | {'Pred Mean':<12}")
        print("  " + "-" * 85)
        
        # Daily RMSE の高い順にソート
        worst = daily.sort_values(by="Daily_RMSE", ascending=False).head(5)
        for idx, (d, row) in enumerate(worst.iterrows()):
            day_of_week = pd.to_datetime(d).strftime('%A')[:3]
            date_str = f"{d} ({day_of_week})"
            print(f"  #{idx+1:<4} | {date_str:<16} | {row['Daily_RMSE']:<12.2f} | {row['Max_Absolute_Error']:<20.2f} | {row['Actual_Mean']:<12.1f} | {row['Pred_Mean']:<12.1f}")

    # 5. グローバルでの最悪日（全モデルの平均が最も高かった日）の直接比較
    # これにより「ゴールデンウィークや特異日における各モデルの真の耐性」を対比できます
    print(f"\n=== [Comparison] Direct Model Comparison on Global Worst Days ===")
    
    # 共通する日付の抽出
    common_dates = None
    for daily in daily_summaries.values():
        if common_dates is None:
            common_dates = set(daily.index)
        else:
            common_dates = common_dates.intersection(set(daily.index))
            
    if common_dates:
        common_dates = sorted(list(common_dates))
        comp_rows = []
        for d in common_dates:
            row = {"Date": d}
            row["DayOfWeek"] = pd.to_datetime(d).strftime('%A')[:3]
            row["Actual_Mean"] = daily_summaries[MODELS[0]].loc[d, "Actual_Mean"]
            
            for model in MODELS:
                if model in daily_summaries:
                    row[f"{model}_RMSE"] = daily_summaries[model].loc[d, "Daily_RMSE"]
            comp_rows.append(row)
            
        comp_df = pd.DataFrame(comp_rows).set_index("Date")
        
        # 全モデルの平均RMSEを計算し、グローバルで難しかった日を特定
        rmse_cols = [f"{m}_RMSE" for m in MODELS if f"{m}_RMSE" in comp_df.columns]
        comp_df["Global_Avg_RMSE"] = comp_df[rmse_cols].mean(axis=1)
        
        # グローバルワースト8日を抽出
        global_worst = comp_df.sort_values(by="Global_Avg_RMSE", ascending=False).head(8)
        
        header_str = f"{'Date (Day)':<16} | {'Actual Mean':<12} | " + " | ".join([f"{MODEL_DISPLAY_NAMES.get(m, m):<12}" for m in MODELS])
        print(header_str)
        print("-" * len(header_str))
        
        for d, row in global_worst.iterrows():
            date_str = f"{d} ({row['DayOfWeek']})"
            val_strs = []
            for m in MODELS:
                col_name = f"{m}_RMSE"
                if col_name in row:
                    val_strs.append(f"{row[col_name]:.2f}")
                else:
                    val_strs.append("N/A")
            print(f"{date_str:<16} | {row['Actual_Mean']:<12.1f} | " + " | ".join([f"{v:<12s}" for v in val_strs]))
            
        # CSVファイルとして保存
        out_csv = THESIS_DIR / "thesis_worst_days_comparison.csv"
        comp_df.to_csv(out_csv, encoding="utf-8-sig")
        print(f"\n[Save] Saved full daily comparison sheet to: {out_csv}")
        
    print("=" * 70)

if __name__ == "__main__":
    analyze_worst_days()
