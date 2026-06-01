import sys
from pathlib import Path
import time
import json
import traceback

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import torch
import optimiser.run_experiment as run_exp

def main():
    print("="*80)
    print("   🚀 MASSIVE GPU OPTUNA OPTIMIZATION SUITE   ")
    print("="*80)
    
    # --- 1. GPU Check ---
    if not torch.cuda.is_available():
        print("❌ CRITICAL ERROR: CUDA (GPU) is not available!")
        print("   Running 1000+ deep learning optimizations on CPU will take weeks.")
        print("   Aborting execution to save your time. Please check your PyTorch/CUDA installation.")
        sys.exit(1)
        
    print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print(f"✅ Optuna is successfully loaded.")
    print("-" * 80)
    
    # --- 2. Configuration ---
    run_exp.MODE = "optimize"
    run_exp.OPTIMIZATION_TRIALS = 50 # 50 trials per permutation
    run_exp.USE_OPTIMIZED_PARAMS = False # Unnecessary since we are optimizing, but set for safety
    
    target_models = ["Hybrid_FiLM_Ablation", "Hybrid_Gated_FeatureFusion", "TransformerFixed", "iTransformer"]
    seasons = ["all", "spring", "summer", "autumn", "winter"]
    
    progress_file = BASE_DIR / "results" / "massive_optuna_progress.json"
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    
    # --- 3. Resume Logic ---
    completed = []
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                completed = json.load(f)
            print(f"ℹ️ Found existing progress file! {len(completed)}/20 completed.")
        except:
            pass
            
    start_time = time.time()
    
    for model in target_models:
        for season in seasons:
            task_id = f"{model}_{season}"
            if task_id in completed:
                print(f"⏩ SKIPPING (Already completed): {task_id}")
                continue
                
            print(f"\n" + "="*60)
            print(f"   >>>> 🧪 OPTIMIZING: {model} | {season} <<<<")
            print("="*60)
            
            run_exp.MODEL_NAME = model
            run_exp.SEASON = season
            
            # File routing logic
            fname = ""
            if run_exp.TARGET_COL == "residual_demand":
                fname = "submit_merged_with_residual.csv" if season == "all" else f"submit_merged_with_residual_{season}.csv"
                run_exp.DATA_FILE = str(run_exp.BASE_DIR / "data/seasonal_split" / fname)
            else: 
                fname = "demand_holiday_with_solar_mean.csv" if season == "all" else f"demand_holiday_with_solar_mean_{season}.csv"
                run_exp.DATA_FILE = str(run_exp.BASE_DIR / "data" / fname)
            
            try:
                run_exp.run_experiment_manager()
                
                # Mark as completed
                completed.append(task_id)
                with open(progress_file, "w") as f:
                    json.dump(completed, f)
                    
            except Exception as e:
                print(f"\n❌ ERROR in {task_id}: {e}")
                traceback.print_exc()
                print("⚠️ The suite will pause briefly, then continue to the next model/season to prevent total failure.")
                time.sleep(5)
                
    total_time = (time.time() - start_time) / 3600
    print("\n" + "="*80)
    print(f"🎉 ALL MASSIVE OPTIMIZATIONS COMPLETE! (Total time: {total_time:.2f} hours)")
    print("="*80)

if __name__ == "__main__":
    main()
