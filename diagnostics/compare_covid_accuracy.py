import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
from pathlib import Path

def compare_accuracy():
    summary_path = Path(r"c:\Users\2213144\practice\result_csv\experiment_summary.csv")
    if not summary_path.exists():
        print("Error: experiment_summary.csv not found.")
        return

    df = pd.read_csv(summary_path)
    
    # Filter for iTransformer and power_demand
    # Note: Older rows might not have 'exclude_covid' or 'target' columns.
    # We assume 'power_demand' if 'target' is missing because 'residual' results are in another folder.
    
    df_it = df[df['model'] == 'iTransformer'].copy()
    
    if len(df_it) < 2:
        print("Not enough iTransformer results to compare.")
        return

    # Let's try to group by 'exclude_covid' if available, otherwise use indices.
    # For now, let's look at the latest runs.
    # We expect 'every' season run to produce 5 rows (spring, summer, autumn, winter, all).
    
    # Grouping by exclusion status
    # If 'exclude_covid' is missing, assume False (default)
    if 'exclude_covid' not in df_it.columns:
        df_it['exclude_covid'] = False
        
    df_it['exclude_covid'] = df_it['exclude_covid'].fillna(False)
    
    inc_df = df_it[df_it['exclude_covid'] == False].tail(5)
    exc_df = df_it[df_it['exclude_covid'] == True].tail(5)
    
    if len(exc_df) == 0:
        print("Exclusion run results not found yet. Please wait for the experiment to finish.")
        return

    print("=== COVID Exclusion Accuracy Comparison (iTransformer) ===")
    print(f"{'Season':<10} | {'Metric':<6} | {'Included':<10} | {'Excluded':<10} | {'Diff (%)':<10}")
    print("-" * 55)
    
    seasons = ["spring", "summer", "autumn", "winter", "all"]
    
    for s in seasons:
        row_inc = inc_df[inc_df['season'] == s]
        row_exc = exc_df[exc_df['season'] == s]
        
        if not row_inc.empty and not row_exc.empty:
            r_inc = row_inc.iloc[0]['rmse']
            r_exc = row_exc.iloc[0]['rmse']
            r_diff = ((r_exc - r_inc) / r_inc) * 100
            
            m_inc = row_inc.iloc[0]['mape']
            m_exc = row_exc.iloc[0]['mape']
            m_diff = ((m_exc - m_inc) / m_inc) * 100
            
            print(f"{s:<10} | {'RMSE':<6} | {r_inc:<10.4f} | {r_exc:<10.4f} | {r_diff:>+9.2f}%")
            print(f"{'':<10} | {'MAPE':<6} | {m_inc:<10.4f} | {m_exc:<10.4f} | {m_diff:>+9.2f}%")
            print("-" * 55)

if __name__ == "__main__":
    compare_accuracy()
