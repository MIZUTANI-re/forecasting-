import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from visualize_models_influence folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Configuration
RESULT_FILE = r"result_csv/experiment_summary.csv"
OUTPUT_DIR = "analysis_results"

# Ensure Output Directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    df.columns = [c.lower() for c in df.columns]
    
    # Keep latest run for each Model + Season
    df = df.drop_duplicates(subset=['model', 'season'], keep='last')
    
    return df

def plot_heatmap(df):
    """Generates a Heatmap of RMSE: Model (Y) vs Season (X)"""
    pivot_df = df.pivot(index='model', columns='season', values='rmse')
    
    # Reorder columns
    seasons_order = ['spring', 'summer', 'autumn', 'winter', 'all']
    existing_cols = [s for s in seasons_order if s in pivot_df.columns]
    pivot_df = pivot_df[existing_cols]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu_r", cbar_kws={'label': 'RMSE (Lower is Better)'})
    plt.title('Model Performance Heatmap (RMSE) - 24h Power Demand Forecast')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap_rmse_24h_power.png")
    print(f"✅ Heatmap saved to {OUTPUT_DIR}/heatmap_rmse_24h_power.png")
    plt.close()

def plot_mape_heatmap(df):
    """Generates a Heatmap of MAPE: Model (Y) vs Season (X)"""
    if 'mape' not in df.columns:
        print("MAPE column not found, skipping.")
        return
        
    pivot_df = df.pivot(index='model', columns='season', values='mape')
    
    seasons_order = ['spring', 'summer', 'autumn', 'winter', 'all']
    existing_cols = [s for s in seasons_order if s in pivot_df.columns]
    pivot_df = pivot_df[existing_cols]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="RdYlGn_r", cbar_kws={'label': 'MAPE % (Lower is Better)'})
    plt.title('Model Performance Heatmap (MAPE) - 24h Power Demand Forecast')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap_mape_24h_power.png")
    print(f"✅ MAPE Heatmap saved to {OUTPUT_DIR}/heatmap_mape_24h_power.png")
    plt.close()

def plot_film_vs_baseline(df):
    """Compare FiLM vs iTransformer"""
    models = ['iTransformer', 'Hybrid_FiLM_Ablation']
    subset = df[df['model'].isin(models)].copy()
    
    if subset.empty:
        print("No matching models for FiLM comparison.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(data=subset, x='season', y='rmse', hue='model', palette='viridis')
    plt.title('24h Power Demand: iTransformer vs Hybrid_FiLM')
    plt.ylabel('RMSE (Lower is Better)')
    plt.xlabel('Season')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/film_vs_itrans_24h_power.png")
    print(f"✅ Comparison plot saved to {OUTPUT_DIR}/film_vs_itrans_24h_power.png")
    plt.close()

def main():
    print(f"📊 Analyzing 24h Power Demand Results from: {RESULT_FILE}")
    df = load_and_clean_data(RESULT_FILE)
    
    if df is not None and not df.empty:
        print(f"Loaded {len(df)} records.")
        plot_heatmap(df)
        plot_mape_heatmap(df)
        plot_film_vs_baseline(df)
        print("✅ 24h Power Demand Analysis Complete.")
    else:
        print("No data to analyze.")

if __name__ == "__main__":
    main()
