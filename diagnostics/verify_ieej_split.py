import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
from optimiser.data_loader import DataLoaderFactory

DATA_FILE = r"C:\Users\2213144\practice\data\demand_holiday_with_solar_mean.csv"
IEEJ_DATES = {
    "train_end": "2020-04-30",
    "test_start": "2021-01-01",
    "test_end": "2021-12-31"
}

def verify():
    print("🚀 Verifying Data Loader with IEEJ Dates...")
    factory = DataLoaderFactory(
        seq_len=72,
        horizon=24,
        strict_split=True,
        sequence_mode="warm",
        target_col="power_demand"
    )
    
    train_loader, val_loader, test_loader, info = factory.load_data(
        file_path=DATA_FILE,
        mode="unified",
        batch_size=64,
        season="all",
        custom_dates=IEEJ_DATES
    )
    
    # Check info
    print("\n✅ Data Loading Ranges (via 'all_timestamp' and 'tr', 'va'):")
    all_ts = info["all_timestamp"]
    tr = info["tr"]
    va = info["va"]
    
    # Train range is 0 to tr
    print(f"Train Range: {all_ts.iloc[0]} TO {all_ts.iloc[tr-1]}")
    
    # Val range is from after training to va
    # Based on data_loader.py: val_df = df.iloc[va_start_idx:va]
    # We can't see va_start_idx easily here without modifying info further, but let's check va
    print(f"Val End:     {all_ts.iloc[va-1]}")
    
    # Test range
    test_starts = info["test_starts"]
    # test sequence starts at s, target starts at s + seq_len
    # last test sequence starts at test_starts[-1], target starts at test_starts[-1] + seq_len
    seq_len = 72
    horizon = 24
    
    first_test_target_start = all_ts.iloc[test_starts[0] + seq_len]
    last_test_target_end = all_ts.iloc[test_starts[-1] + seq_len + horizon - 1]
    
    print(f"Test Target (First): {first_test_target_start}")
    print(f"Test Target (Last):  {last_test_target_end}")

if __name__ == "__main__":
    verify()
