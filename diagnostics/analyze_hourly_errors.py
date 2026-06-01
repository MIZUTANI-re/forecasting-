import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ============================================================
# Font & Design Settings (Premium & Paper-readable / Academic Thesis Style)
# ============================================================
plt.rcParams["axes.unicode_minus"] = False
JP_FONTS = ["Yu Gothic", "Yu Gothic UI", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"]
plt.rcParams["font.family"] = JP_FONTS

# --- Style Options ---
DPI = 600
SAVE_KW = dict(dpi=DPI, bbox_inches="tight", pad_inches=0.04)

FONT_TITLE  = 13
FONT_LABEL  = 11
FONT_TICK   = 9
FONT_LEGEND = 9

FIG_W = 7.2
FIG_H_MAIN = 4.0
FIG_H_ERROR = 3.6
FIG_H_BOX = 4.2

# 美しく洗練されたデフォルトカラー
DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# プログラム上の変数名から、論文・スライド掲載用のクリーンな名前へのマッピング
MODEL_DISPLAY_NAMES = {
    "TransformerFixed": "Transformer",
    "iTransformer": "iTransformer",
    "Hybrid_Predict_Fusion": "Hybrid",
}

def get_model_color(model_name, idx):
    """
    モデル名に基づいて一貫したカラーを割り当てます。
    【重要】予測グラフと誤差(Bias)グラフで、モデルの色を完全に統一します！
    - Actual: ブルー (#1f77b4)
    - TransformerFixed / Transformer: オレンジ (#ff7f0e)
    - iTransformer: グリーン (#2ca02c)
    - Hybrid: レッド (#d62728)
    """
    fixed_colors = {
        "actual": "#1f77b4",
        "itransformer": "#2ca02c",
        "transformerfixed": "#ff7f0e",
        "transformer": "#ff7f0e",
        "hybrid": "#d62728",
    }
    name_lower = model_name.lower()
    for k, v in fixed_colors.items():
        if k in name_lower:
            return v
    if "hybrid" in name_lower:
        return "#d62728"
    return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

# Line styles and markers (お手本グラフのデザインを忠実に再現)
LW_ACTUAL = 3.2  # 実測値は太い線で強調
LW_MODEL  = 2.0
MS_ACTUAL = 5.0
MS_MODEL  = 4.0
MEW       = 0.8

# ============================================================
# Directory Settings
# ============================================================
BASE_DIR   = Path(r"c:\Users\2213144\practice")
RESULT_DIR = BASE_DIR / "result_csv"

# 論文用に特化した専用の保存フォルダ 'thesis'
THESIS_DIR = BASE_DIR / "thesis"
THESIS_DIR.mkdir(parents=True, exist_ok=True)

# 詳細ログ・デバッグ用の画像出力フォルダ
OUT_DIR    = BASE_DIR / "picture" / "hourly_error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 季節の定義
SEASONS = ["spring", "summer", "autumn", "winter", "all"]

# 比較対象のモデルフォルダ名のリスト
MODELS  = [
    "TransformerFixed", 
    "iTransformer", 
    "Hybrid_Predict_Fusion"
]

SCALE = 1.0
UNIT_EN = "10^4 kW"

# ============================================================
# Helper Functions
# ============================================================
def read_csv_auto(path) -> pd.DataFrame:
    path = str(path)
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    return pd.read_csv(path)

def resolve_prediction_paths():
    paths = {model: {} for model in MODELS}
    
    for model in MODELS:
        model_dir = RESULT_DIR / model
        if not model_dir.exists():
            matched_dirs = [d for d in RESULT_DIR.iterdir() if d.is_dir() and d.name.lower() == model.lower()]
            if matched_dirs:
                model_dir = matched_dirs[0]
            else:
                print(f"[Warning] Model directory not found: {model_dir}")
                for s in SEASONS:
                    paths[model][s] = None
                continue
                
        for s in SEASONS:
            patterns = [
                f"*_{s}_unified.csv",
                f"*_{s}_*.csv",
                f"*{s}*.csv"
            ]
            
            found_file = None
            for pat in patterns:
                files = list(model_dir.glob(pat))
                if files:
                    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    found_file = str(files[0])
                    break
            paths[model][s] = found_file
            
    return paths

# ============================================================
# Core Analysis & Data Loading
# ============================================================
def load_pred_data_full(path: str, season_label: str) -> pd.DataFrame:
    df = read_csv_auto(path)
    
    time_col = None
    for col in ["timestamp", "datetime", "date", "target_datetime"]:
        if col in df.columns:
            time_col = col
            break
            
    if time_col is None:
        raise ValueError(f"{path}: timestamp/datetime column missing. Columns: {list(df.columns)}")
    
    for col in ["y_true", "y_pred"]:
        if col not in df.columns:
            raise ValueError(f"{path}: {col} column missing")
            
    df[time_col] = pd.to_datetime(df[time_col].astype(str).str.strip(), errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=[time_col, "y_true", "y_pred"]).copy()
    df = df.rename(columns={time_col: "target_datetime"})
    
    if season_label != "all" and "season" in df.columns and df["season"].notna().any():
        df = df[df["season"].astype(str) == season_label].copy()
        
    df["hour"] = df["target_datetime"].dt.hour
    return df[["target_datetime", "hour", "y_true", "y_pred"]]

def analyze_hourly_errors():
    paths = resolve_prediction_paths()
    
    print("\n=== Resolved Prediction CSV Files ===")
    for model in MODELS:
        print(f"Model: {model}")
        for season in SEASONS:
            p = paths[model][season]
            p_name = Path(p).name if p else "NOT FOUND"
            print(f"  - {season:7s} -> {p_name}")
            
    for season in SEASONS:
        print(f"\n========================================\n[Processing Season] {season}\n========================================")
        model_dfs = {}
        
        for model in MODELS:
            p = paths[model][season]
            if p:
                try:
                    model_dfs[model] = load_pred_data_full(p, season_label=season)
                except Exception as e:
                    print(f"Error loading {model} for {season}: {e}")
                    model_dfs[model] = None
            else:
                model_dfs[model] = None
                
        if not any(df is not None for df in model_dfs.values()):
            print(f"No prediction data found for season: {season}. Skipping.")
            continue
            
        stats_summary = []
        error_dist_by_model = {m: [[] for _ in range(24)] for m in MODELS if model_dfs[m] is not None}
        
        for hour in range(24):
            hour_stats = {"Hour": hour}
            y_true_vals = []
            for model in MODELS:
                df = model_dfs[model]
                if df is not None:
                    hour_df = df[df["hour"] == hour]
                    if not hour_df.empty:
                        y_true_vals.append(hour_df["y_true"].values)
                        
            if y_true_vals:
                actual_mean = np.mean(y_true_vals[0]) / SCALE
                hour_stats["Actual (Test Data)"] = actual_mean
            else:
                hour_stats["Actual (Test Data)"] = np.nan
                
            for model in MODELS:
                df = model_dfs[model]
                if df is not None:
                    hour_df = df[df["hour"] == hour]
                    if not hour_df.empty:
                        y_t = hour_df["y_true"].values / SCALE
                        y_p = hour_df["y_pred"].values / SCALE
                        errors = y_p - y_t
                        
                        mbe = np.mean(errors)
                        mae = np.mean(np.abs(errors))
                        rmse = np.sqrt(np.mean(errors**2))
                        
                        hour_stats[f"{model}_Pred_Mean"] = np.mean(y_p)
                        hour_stats[f"{model}_MBE"] = mbe
                        hour_stats[f"{model}_MAE"] = mae
                        hour_stats[f"{model}_RMSE"] = rmse
                        
                        error_dist_by_model[model][hour] = errors
                    else:
                        hour_stats[f"{model}_Pred_Mean"] = np.nan
                        hour_stats[f"{model}_MBE"] = np.nan
                        hour_stats[f"{model}_MAE"] = np.nan
                        hour_stats[f"{model}_RMSE"] = np.nan
                        
            stats_summary.append(hour_stats)
            
        stats_df = pd.DataFrame(stats_summary).set_index("Hour")
        
        # 統計データを thesis と picture の両フォルダに保存
        stats_df.to_csv(THESIS_DIR / f"thesis_hourly_metrics_{season}.csv", encoding="utf-8-sig")
        stats_df.to_csv(OUT_DIR / f"hourly_error_metrics_{season}.csv", encoding="utf-8-sig")
        print(f"Saved CSV metrics: thesis_hourly_metrics_{season}.csv")
        
        # ============================================================
        # 📊 グラフ1: 予測値 vs 実測テストデータ (英語表記のみ)
        # ============================================================
        fig_en, ax_en = plt.subplots(figsize=(FIG_W + 0.3, FIG_H_MAIN), dpi=DPI)
        
        if not stats_df["Actual (Test Data)"].isna().all():
            ax_en.plot(stats_df.index, stats_df["Actual (Test Data)"], label="Actual (Test Data)", 
                       color=get_model_color("actual", 0), linewidth=LW_ACTUAL, marker="o", markersize=MS_ACTUAL, zorder=3)
                     
        for idx, model in enumerate(MODELS):
            col = f"{model}_Pred_Mean"
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            if col in stats_df.columns and not stats_df[col].isna().all():
                ax_en.plot(stats_df.index, stats_df[col], label=display_name, 
                           color=get_model_color(model, idx), linewidth=LW_MODEL, marker="o", markersize=MS_MODEL, zorder=2)
                         
        ax_en.set_xticks(range(24))
        ax_en.set_xlabel("Hour of Day", fontsize=FONT_LABEL)
        ax_en.set_ylabel(f"Average Demand ({UNIT_EN})", fontsize=FONT_LABEL)
        ax_en.set_title(f"Hourly Average Demand Comparison - Test Period (Season: {season})", fontsize=FONT_TITLE, fontweight="bold")
        ax_en.tick_params(axis="both", labelsize=FONT_TICK)
        
        # 'all' シーズンの需要の縦軸範囲を 220〜360、10刻みに固定
        if season == "all":
            ax_en.set_ylim(220, 360)
            ax_en.yaxis.set_major_locator(mticker.MultipleLocator(10))
            
        ax_en.grid(True, which="both", linestyle="-", color="#e0e0e0", alpha=0.8)
        ax_en.legend(loc="upper left", frameon=False, fontsize=FONT_LEGEND)
        plt.tight_layout()
        
        fig_en.savefig(THESIS_DIR / f"thesis_hourly_demand_{season}.png", **SAVE_KW)
        fig_en.savefig(OUT_DIR / f"hourly_demand_{season}.png", **SAVE_KW)
        plt.close(fig_en)

        print(f"Saved: thesis_hourly_demand_{season}.png")

        # ============================================================
        # 📈 グラフ2: 平均予測バイアス (Bias = Pred - Actual)
        # 【重要】縦軸は電力需要と同じ物理単位、かつ予測グラフと「色を完全統一」！
        # ============================================================
        fig_bias, ax_bias = plt.subplots(figsize=(FIG_W + 0.3, FIG_H_ERROR), dpi=DPI)
        
        ax_bias.axhline(0, color="black", linewidth=1.2, linestyle="-", zorder=1)
        
        for idx, model in enumerate(MODELS):
            col = f"{model}_MBE"
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            if col in stats_df.columns and not stats_df[col].isna().all():
                # 予測グラフと100%同じ色(get_model_color)を使用して描画！
                ax_bias.plot(stats_df.index, stats_df[col], label=display_name, 
                             color=get_model_color(model, idx), linewidth=LW_MODEL, marker="s", markersize=MS_MODEL, linestyle="-", zorder=2)
                             
        ax_bias.set_xticks(range(24))
        ax_bias.set_xlabel("Hour of Day", fontsize=FONT_LABEL)
        ax_bias.set_ylabel(f"Bias (Pred - Actual) ({UNIT_EN})", fontsize=FONT_LABEL)
        ax_bias.set_title(f"Hourly Forecast Bias - Test Period (Season: {season})", fontsize=FONT_TITLE, fontweight="bold")
        ax_bias.tick_params(axis="both", labelsize=FONT_TICK)
        
        # 縦軸のスケールを電力需要と同じ単位の差分（-20 〜 20、5刻み）に統一
        ax_bias.set_ylim(-20, 20)
        ax_bias.yaxis.set_major_locator(mticker.MultipleLocator(5))
        
        ax_bias.grid(True, which="both", linestyle="-", color="#e0e0e0", alpha=0.8)
        ax_bias.legend(loc="lower left", frameon=False, fontsize=FONT_LEGEND)
        plt.tight_layout()
        
        fig_bias.savefig(THESIS_DIR / f"thesis_hourly_bias_{season}.png", **SAVE_KW)
        fig_bias.savefig(OUT_DIR / f"hourly_bias_{season}.png", **SAVE_KW)
        plt.close(fig_bias)

        print(f"Saved: thesis_hourly_bias_{season}.png")

        # ============================================================
        # 📊 グラフ3: 誤差のばらつき分布 (Boxplot) (予測グラフと色を完全統一)
        # ============================================================
        for idx, model in enumerate(MODELS):
            if model not in error_dist_by_model or not any(len(x) > 0 for x in error_dist_by_model[model]):
                continue
                
            fig, ax = plt.subplots(figsize=(FIG_W + 0.8, FIG_H_BOX))
            
            box_data = [error_dist_by_model[model][h] for h in range(24)]
            box_data = [x if len(x) > 0 else [0] for x in box_data]
            
            # 予測・バイアスグラフと完全に統一された色を取得
            model_color = get_model_color(model, idx)
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            
            bp = ax.boxplot(
                box_data, 
                positions=range(24), 
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker="^", markerfacecolor="red", markeredgecolor="red", markersize=5),
                flierprops=dict(marker=".", markerfacecolor="gray", markersize=3, alpha=0.4, linestyle="none"),
                boxprops=dict(facecolor=model_color + "33", color=model_color, linewidth=1.2),
                whiskerprops=dict(color=model_color, linewidth=1.0),
                capprops=dict(color=model_color, linewidth=1.0),
                medianprops=dict(color="black", linewidth=1.2)
            )
            
            ax.axhline(0, color="black", linewidth=1.2, linestyle="-")
            ax.set_xticks(range(24))
            ax.set_xticklabels(range(24))
            ax.set_xlabel("Hour of Day", fontsize=FONT_LABEL)
            ax.set_ylabel(f"Forecast Error (Pred - Actual) ({UNIT_EN})", fontsize=FONT_LABEL)
            ax.set_title(f"Hourly Forecast Error Distribution - Test Period: {display_name}\n(Season: {season}, Red ▲ = Mean)", fontsize=FONT_TITLE, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6)
            
            plt.tight_layout()
            
            fig.savefig(THESIS_DIR / f"thesis_error_distribution_{display_name}_{season}.png", **SAVE_KW)
            fig.savefig(OUT_DIR / f"hourly_error_distribution_{display_name}_{season}.png", **SAVE_KW)
            plt.close(fig)

if __name__ == "__main__":
    analyze_hourly_errors()
    print("\n========================================")
    print("ALL TIME-OF-DAY ERROR ANALYSIS COMPLETED FOR THESIS!")
    print(f"Thesis Directory (Clean): {THESIS_DIR}")
    print(f"Output Directory (Detail): {OUT_DIR}")
    print("========================================")
