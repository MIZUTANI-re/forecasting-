
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

def generate_individual_yearly_plots():
    # Paths
    base_dir = Path(r"c:\Users\2213144\practice")
    data_path = base_dir / "data" / "demand_holiday_with_solar_mean.csv"
    save_dir = base_dir / "picture" / "yearly_plots_individual"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Preprocessing
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    
    years = sorted(df['year'].unique())
    print(f"📊 Generating plots for years: {years}")

    for year in years:
        year_data = df[df['year'] == year].copy()
        
        # 1. Full Year Time Series Plot
        plt.figure(figsize=(15, 6))
        plt.plot(year_data['datetime'], year_data['power_demand'], color='tab:blue', linewidth=0.8)
        plt.title(f"Power Demand Time Series - Year {year}")
        plt.xlabel("Datetime")
        plt.ylabel("Power Demand")
        plt.grid(True, alpha=0.3)
        
        # If 2024, adjust x-limits to the actual data range to make it visible
        if year == 2024:
            plt.xlim(year_data['datetime'].min(), year_data['datetime'].max())
            plt.title(f"Power Demand Time Series - Year {year} (Zoomed to available data)")
        
        # Save plot
        plot_name = f"demand_timeseries_{year}.png"
        plt.savefig(save_dir / plot_name)
        plt.close()
        print(f"✅ Generated: {plot_name}")

        # 2. Average Daily Profile for this specific year
        # Use the 'time' column which represents the hour
        daily_profile = year_data.groupby('time')['power_demand'].mean()
        
        plt.figure(figsize=(10, 5))
        plt.plot(daily_profile.index, daily_profile.values, marker='o', color='tab:red', linestyle='-', linewidth=2)
        plt.title(f"Average Daily Demand Profile - Year {year}")
        plt.xlabel("Hour of Day")
        plt.ylabel("Average Power Demand")
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        
        profile_name = f"daily_profile_{year}.png"
        plt.savefig(save_dir / profile_name)
        plt.close()
        print(f"✅ Generated: {profile_name}")

    print(f"\n✨ All plots saved to: {save_dir}")

if __name__ == "__main__":
    generate_individual_yearly_plots()
