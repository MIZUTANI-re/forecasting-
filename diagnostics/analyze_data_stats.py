import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
import os
import sys

# Standard Season Definition
# Spring: 3,4,5
# Summer: 6,7,8
# Autumn: 9,10,11
# Winter: 12,1,2

def get_season(month):
    if month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "autumn"
    else:
        return "winter"

def analyze_stats(file_path):
    print(f"📊 Analyzing Data Statistics for: {file_path}")
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    df = pd.read_csv(file_path)
    
    time_col = "datetime" if "datetime" in df.columns else "timestamp"
    if time_col in df.columns:
        df["timestamp"] = pd.to_datetime(df[time_col])
        df["month"] = df["timestamp"].dt.month
        df["season"] = df["month"].apply(get_season)
    else:
        print("⚠ 'timestamp' column not found. Unable to determine seasons automatically.")
        return

    # Check for target column (adjust if your target name is different)
    # Common names: "demand", "value", "target", "OT"
    # trying to detect numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude helper columns
    exclude = ["month", "hour", "minute", "day", "year"]
    target_candidates = [c for c in numeric_cols if c not in exclude]
    
    if not target_candidates:
        print("❌ No target variable found.")
        return
    
    # Assume first candidate is target or let user specify. 
    # Based on previous context, it is likely "demand" or the last column.
    # We will analyze ALL numerical columns just in case, or focus on the main one.
    # Let's pick the last one as per standard ETT/Energy datasets often having target at end.
    # Prefer 'power_demand' if present, else last column
    if "power_demand" in target_candidates:
        target_col = "power_demand"
    else:
        target_col = target_candidates[-1] 
    print(f"🎯 Target Variable: {target_col}\n")
    
    seasons = ["all", "spring", "summer", "autumn", "winter"]
    
    print(f"{'Season':<10} | {'Count':<10} | {'Mean':<12} | {'Std Dev':<12} | {'Min':<10} | {'Max':<10}")
    print("-" * 80)
    
    for s in seasons:
        if s == "all":
            data = df[target_col]
        else:
            data = df[df["season"] == s][target_col]
            
        if len(data) == 0:
            print(f"{s:<10} | {'0':<10} | {'-':<12} | {'-':<12}")
            continue
            
        count = len(data)
        mean = data.mean()
        std = data.std()
        min_v = data.min()
        max_v = data.max()
        
        print(f"{s:<10} | {count:<10} | {mean:<12.4f} | {std:<12.4f} | {min_v:<10.2f} | {max_v:<10.2f}")

if __name__ == "__main__":
    # Default path based on previous context
    default_path = r"C:\Users\2213144\practice\data\demand_holiday_with_solar_mean.csv"
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = default_path
        
    analyze_stats(path)
