import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = r"C:\Users\2213144\practice\data\demand_holiday_with_solar_mean.csv"
TARGET_HOURS = range(11, 21) # 11 to 20

def analyze_daytime_shift():
    print(f"📊 Analyzing Demand Distribution for Hours {list(TARGET_HOURS)}...")
    
    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    if "time" not in df.columns or "power_demand" not in df.columns:
        print("❌ Dataset missing columns.")
        return

    # 2. Split (approximate based on standard 8:1:1 or just use last day)
    # The last 24h is the target of interest.
    full_len = len(df)
    train_len = int(full_len * 0.8)
    
    train_df = df.iloc[:train_len]
    # Assuming the "last 24h plot" corresponds to the very end of the dataset
    test_target_df = df.tail(24) 
    
    # 3. Filter for target hours
    train_daytime = train_df[train_df["time"].isin(TARGET_HOURS)]["power_demand"]
    test_daytime  = test_target_df[test_target_df["time"].isin(TARGET_HOURS)]["power_demand"]
    
    # 4. Statistics
    tr_mean = train_daytime.mean()
    tr_std = train_daytime.std()
    
    te_mean = test_daytime.mean()
    te_values = test_daytime.values
    
    print("\n--- Training Set (80%) Stats (11:00 - 20:00) ---")
    print(f"Mean Demand: {tr_mean:.2f}")
    print(f"Std Dev:     {tr_std:.2f}")
    print(f"Range:       [{train_daytime.min():.2f}, {train_daytime.max():.2f}]")
    
    print("\n--- Last 24 Hours Stats (11:00 - 20:00) ---")
    print(f"Mean Demand: {te_mean:.2f}")
    print(f"Actual Values: {te_values}")
    
    diff = te_mean - tr_mean
    sigma_diff = diff / tr_std if tr_std > 0 else 0
    
    print("\n--- Comparison ---")
    print(f"Difference (Test - Train Mean): {diff:.2f}")
    print(f"Deviation (in Sigma):           {sigma_diff:.2f} σ")
    
    if abs(sigma_diff) > 1.0:
        print("❗ SIGNIFICANT SHIFT: The test day demand is notably different from the training average.")
    else:
        print("✅ Distribution seems consistent.")
        
    # Check Solar if available
    if "solar_mean" in df.columns:
        train_solar = train_df[train_df["time"].isin(TARGET_HOURS)]["solar_mean"].mean()
        test_solar = test_target_df[test_target_df["time"].isin(TARGET_HOURS)]["solar_mean"].mean()
        print(f"\n☀️ Solar Mean (Feature) Comparison (11-20h):")
        print(f"Train Solar: {train_solar:.2f}")
        print(f"Test Solar:  {test_solar:.2f}")

if __name__ == "__main__":
    analyze_daytime_shift()
