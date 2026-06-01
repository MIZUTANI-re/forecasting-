
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def analyze_yearly_profiles():
    # Paths
    base_dir = Path(r"c:\Users\2213144\practice")
    data_path = base_dir / "data" / "demand_holiday_with_solar_mean.csv"
    save_dir = base_dir / "picture" / "yearly_comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Preprocessing
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['timestamp'].dt.month
    
    years = sorted(df['year'].unique())
    print(f"📊 Years found: {years}")

    # 1. Yearly Demand Profile (Average Day)
    plt.figure(figsize=(12, 6))
    for year in years:
        yearly_profile = df[df['year'] == year].groupby('hour')['power_demand'].mean()
        # Highlight COVID years (2020-2022) with different style
        style = '--' if 2020 <= year <= 2022 else '-'
        linewidth = 3 if 2020 <= year <= 2022 else 1.5
        plt.plot(yearly_profile.index, yearly_profile.values, label=f"{year}", linestyle=style, linewidth=linewidth)
    
    plt.title("Yearly Average Daily Demand Profile (2016-2024)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Power Demand (average)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / "yearly_daily_profiles.png")
    print(f"✅ Saved Profile Plot: {save_dir / 'yearly_daily_profiles.png'}")

    # 2. Demand vs Temperature Scatter (Yearly Comparison)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, year in enumerate(years):
        if i >= len(axes): break
        year_data = df[df['year'] == year]
        sns.regplot(data=year_data, x='temperature', y='power_demand', 
                    ax=axes[i], scatter_kws={'alpha': 0.1, 's': 1}, line_kws={'color': 'red'})
        axes[i].set_title(f"Year {year}")
        axes[i].grid(True, alpha=0.2)
        
    plt.suptitle("Demand vs Temperature Correlation per Year", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_dir / "yearly_temp_correlation.png")
    print(f"✅ Saved Correlation Plot: {save_dir / 'yearly_temp_correlation.png'}")

    # 3. Basic Stats Table per Year
    stats = df.groupby('year')['power_demand'].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
    stats['range'] = stats['max'] - stats['min']
    
    stats_csv = save_dir / "yearly_stats_summary.csv"
    stats.to_csv(stats_csv)
    print(f"✅ Saved Stats CSV: {stats_csv}")
    
    print("\n--- Yearly Stats Summary ---")
    print(stats)

    # 4. Boxplot comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='year', y='power_demand')
    plt.title("Power Demand Distribution per Year")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_dir / "yearly_demand_boxplot.png")
    print(f"✅ Saved Boxplot: {save_dir / 'yearly_demand_boxplot.png'}")

if __name__ == "__main__":
    analyze_yearly_profiles()
