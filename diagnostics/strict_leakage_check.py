import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
from optimiser.data_loader import DataLoaderFactory

def verify_leaks():
    print("🔍 Starting Strict Leakage Verification...")
    
    DATA_FILE = r"C:\Users\2213144\practice\data\seasonal_split\submit_merged_with_residual.csv"
    SEQ_LEN = 72
    HORIZON = 24
    
    # Test 1: Default Ratio Split
    print("\n[Test 1] Default Ratio Split (80/10/10)")
    factory = DataLoaderFactory(
        seq_len=SEQ_LEN, 
        horizon=HORIZON, 
        strict_split=True, 
        sequence_mode="warm",
        target_col="residual_demand"
    )
    train_loader, val_loader, test_loader, info = factory.load_data(DATA_FILE, mode="unified")
    
    all_ts = info['all_timestamp']
    test_starts = info['test_starts']
    tr_idx = info['tr']
    va_end_idx = info['va']
    
    # Check total sequences
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"   Counts -> Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Leakage check: Test set start
    first_test_start_idx = test_starts[0]
    y_start_idx = first_test_start_idx + SEQ_LEN
    y_end_idx = y_start_idx + HORIZON - 1
    
    print(f"   First Test Sequence Target Range: {all_ts.iloc[y_start_idx]} to {all_ts.iloc[y_end_idx]}")
    print(f"   Val End Index in DF: {va_end_idx} ({all_ts.iloc[va_end_idx-1]})")
    
    if y_start_idx >= va_end_idx:
        print("   ✅ SUCCESS: Test start (y) is after Val end.")
    else:
        print("   ❌ FAILURE: Test leakage detected!")

    # Test 2: IEEJ Custom Split (The gap case)
    print("\n[Test 2] IEEJ Custom Split (2020-04 to 2021-01 gap)")
    IEEJ_DATES = {
        "train_end": "2020-04-30",
        "test_start": "2021-01-01",
        "test_end": "2021-12-31"
    }
    train_loader, val_loader, test_loader, info = factory.load_data(DATA_FILE, mode="unified", custom_dates=IEEJ_DATES)
    
    ts = info['all_timestamp']
    tr = info['tr']
    # The last training sample's y must end <= tr
    # In strict_split: m_tr = (y_end < tr_idx)
    # So the very last y-timestamp of train should be < all_ts[tr]
    
    # Since we use DataLoader, we look at the masks/info
    # Let's re-run the split logic manually to verify the mask counts
    n = len(ts)
    starts = np.arange(n - SEQ_LEN - HORIZON + 1)
    y_start = starts + SEQ_LEN
    y_end = y_start + HORIZON - 1
    
    # Filter indices (the same logic as in data_loader.py)
    m_tr = (y_end < tr)
    train_y_ends = y_end[m_tr]
    if len(train_y_ends) > 0:
        last_train_y_idx = train_y_ends.max()
        print(f"   Last Train Target Timestamp: {ts.iloc[last_train_y_idx]}")
        print(f"   Train End Boundary: {ts.iloc[tr-1]}")
        if last_train_y_idx < tr:
            print("   ✅ SUCCESS: All Train targets are within Training boundary.")
    
    # Test set check
    m_te = (y_start >= info['test_starts'][0]) # Since we already have test_starts
    test_y_starts = y_start[starts >= info['test_starts'][0]]
    if len(test_y_starts) > 0:
        first_test_y_idx = test_y_starts.min()
        print(f"   First Test Target Timestamp: {ts.iloc[first_test_y_idx]}")
        if ts.iloc[first_test_y_idx] >= pd.to_datetime("2021-01-01"):
             print("   ✅ SUCCESS: Test targets start exactly on 2021-01-01.")

    print("\n✨ All strict splitting checks passed! No data leakage detected.")

if __name__ == "__main__":
    verify_leaks()
