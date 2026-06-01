import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from diagnostics folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
from pathlib import Path

def analyze_covid_reduction():
    file_path = r"C:\Users\2213144\practice\data\seasonal_split\submit_merged_with_residual.csv"
    if not Path(file_path).exists():
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    total_rows = len(df)
    
    # COVID Period: 2020-02-01 to 2022-12-31
    covid_start = pd.to_datetime("2020-02-01")
    covid_end = pd.to_datetime("2022-12-31")
    
    covid_df = df[(df['datetime'] >= covid_start) & (df['datetime'] <= covid_end)]
    covid_rows = len(covid_df)
    
    reduction_pct = (covid_rows / total_rows) * 100
    
    print(f"--- Data Reduction Analysis ---")
    print(f"Total Rows: {total_rows}")
    print(f"COVID Period Rows (2020/02 - 2022/12): {covid_rows}")
    print(f"Remaining Rows: {total_rows - covid_rows}")
    print(f"Exclusion Percentage: {reduction_pct:.2f}%")
    
    # Check by year to show the distribution
    df['year'] = df['datetime'].dt.year
    print("\nRows per Year:")
    print(df['year'].value_counts().sort_index())

if __name__ == "__main__":
    analyze_covid_reduction()
