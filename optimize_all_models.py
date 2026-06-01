
import sys
from pathlib import Path
import time
import traceback

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import optimiser.run_experiment as run_exp

# 13 Models
MODELS = [
    "FiLM", 
    "iTransformer", 
    "TransformerFixed", 
    "Hybrid_FiLM_Ablation",
    "Hybrid_Predict_Fusion", 
    "Hybrid_NoHQ_Predict_Fusion", 
    "Hybrid_Gated_FeatureFusion", 
    "Hybrid_Gated_QueryFusion", 
    "Hybrid_KAN_Gated_FeatureFusion", 
    "htmformer", 
    "MLP", 
    "grid_tst", 
    "MMK"
]

SEASONS = ["spring", "summer", "autumn", "winter", "all"]

def main():
    print("🚀 STARTING FULL OPTIMIZATION SUITE")
    print(f"Models: {len(MODELS)} | Seasons: {len(SEASONS)}")
    print("-" * 50)
    
    # Global Settings
    run_exp.MODE = "optimize"
    run_exp.OPTIMIZATION_TRIALS = 10 
    
    start_time = time.time()
    
    for model in MODELS:
        for season in SEASONS:
            print(f"\n>>>> OPTIMIZING: {model} | {season} <<<<")
            try:
                # Set Globals
                run_exp.MODEL_NAME = model
                run_exp.SEASON = season
                
                # Set Data File Logic
                fname = ""
                if run_exp.TARGET_COL == "residual_demand":
                    fname = "submit_merged_with_residual.csv" if season == "all" else f"submit_merged_with_residual_{season}.csv"
                    run_exp.DATA_FILE = str(run_exp.BASE_DIR / "data/seasonal_split" / fname)
                else: 
                    fname = "demand_holiday_with_solar_mean.csv" if season == "all" else f"demand_holiday_with_solar_mean_{season}.csv"
                    run_exp.DATA_FILE = str(run_exp.BASE_DIR / "data" / fname)
                
                # Run Manager
                run_exp.run_experiment_manager()
                
            except Exception as e:
                print(f"❌ FAILED {model} - {season}")
                print(traceback.format_exc())
                # Continue to next
            
            print(f"---- Finished {model} - {season} ----\n")
            
    total_time = (time.time() - start_time) / 3600
    print(f"✅ FULL OPTIMIZATION COMPLETE in {total_time:.2f} hours")

if __name__ == "__main__":
    main()
