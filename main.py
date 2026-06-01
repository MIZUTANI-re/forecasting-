
import os
import sys
from pathlib import Path

# Add project root to sys.path to ensure modules are found
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from optimiser.run_experiment import main as run_pipeline, HP, SEASON, DATA_FILE, SEQ_LEN, HORIZON, EPOCHS, BATCH_SIZE, LEARNING_RATE
import optimiser.run_experiment as runner_config

def interactive_main():
    print("="*60)
    print("   🚀 Practice Model Runner   ")
    print("="*60)
    
    # --- ADDED: OPTIMIZATION OVERRIDE LOGIC ---
    opt_choice = input("\n過去に最適化されたベストパラメータを使用して、特定の4モデル×5季節の分析を自動実行しますか？ (yes/no): ").strip().lower()
    if opt_choice in ['y', 'yes']:
        import time
        import traceback
        
        runner_config.USE_OPTIMIZED_PARAMS = True
        runner_config.MODE = "fixed"
        runner_config.OPTIMIZATION_TRIALS = 1
        
        target_models = ["Hybrid_FiLM_Ablation", "Hybrid_Gated_FeatureFusion", "TransformerFixed", "iTransformer"]
        seasons = ["all", "spring", "summer", "autumn", "winter"]
        
        print(f"\n✅ 最適化パラメータを使用し、以下の分析を実行します。")
        print(f"Models: {target_models}")
        print(f"Seasons: {seasons}\n")
        
        start_time = time.time()
        for model in target_models:
            for season in seasons:
                print(f"\n>>>> EXECUTING: {model} | {season} <<<<")
                runner_config.MODEL_NAME = model
                runner_config.SEASON = season
                
                fname = ""
                if runner_config.TARGET_COL == "residual_demand":
                    fname = "submit_merged_with_residual.csv" if season == "all" else f"submit_merged_with_residual_{season}.csv"
                    runner_config.DATA_FILE = str(runner_config.BASE_DIR / "data/seasonal_split" / fname)
                else: 
                    fname = "demand_holiday_with_solar_mean.csv" if season == "all" else f"demand_holiday_with_solar_mean_{season}.csv"
                    runner_config.DATA_FILE = str(runner_config.BASE_DIR / "data" / fname)
                
                try:
                    runner_config.run_experiment_manager()
                except Exception as e:
                    print(f"❌ ERROR in {model} - {season}: {e}")
                    traceback.print_exc()
                    
        total_time = (time.time() - start_time) / 3600
        print(f"\n✅ 全ての分析が完了しました (所要時間: {total_time:.2f}時間)")
        return  # Exit main.py since the requested suite has completed.

    else:
        runner_config.USE_OPTIMIZED_PARAMS = False
        print("\nℹ️ 通常モードで開始します。")
    # ------------------------------------------

    print("\n[Configuration Reference]")
    
    # Target Selection
    print("\nSelect Target Variable:")
    print("  1. power_demand (Default)")
    print("  2. residual_demand")
    t_choice = input("Enter number (1-2): ").strip()
    target_col = "residual_demand" if t_choice == "2" else "power_demand"
    runner_config.TARGET_COL = target_col
    print(f"✅ Target Set to: {target_col}")

    print(f"✅ Target Set to: {target_col}")

    # Data Source Selection (Auto-determined)
    if target_col == "residual_demand":
        use_seasonal_files = True
        print("✅ Data Source: Seasonal Split Files (Auto-selected for Residual Demand)")
    else:
        use_seasonal_files = False
        print("✅ Data Source: Original Unified File (Auto-selected for Power Demand)")

    # ------------------------------------------------------------
    # 2. Season Selection
    # ------------------------------------------------------------
    seasons = ["all", "spring", "summer", "autumn", "winter", "every"]
    print("\nSelect season (default: all):")
    for i, s in enumerate(seasons):
        print(f"  {i+1}. {s}")
        
    choice_s = input("Enter number (Enter for default): ").strip()
    selected_season = "all"
    if choice_s:
        try:
            s_idx = int(choice_s) - 1
            if 0 <= s_idx < len(seasons):
                selected_season = seasons[s_idx]
        except ValueError:
            pass
            
    print(f"✅ Selected Season: {selected_season}")
    runner_config.SEASON = selected_season

    # ------------------------------------------------------------
    # 3. Horizon Selection
    # ------------------------------------------------------------
    print("\nSelect Prediction Horizon:")
    print("  1. 24 hours (Default)")
    print("  2. 72 hours")
    h_choice = input("Enter number (1-2): ").strip()
    if h_choice == '2':
        runner_config.HORIZON = 72
        # Usually Pre-len matches Horizon, so update that too just in case config uses it
        # (Though models often take 'pred_len' from config)
        # runner_config.HORIZON is used in run_experiment
    else:
        runner_config.HORIZON = 24
        
    print(f"✅ Horizon Set to: {runner_config.HORIZON} hours")

    # ------------------------------------------------------------
    # 4. Model Selection
    # ------------------------------------------------------------

    models = [
        "iTransformer",
        "grid_tst", 
        "Hybrid_Gated_FeatureFusion",
        "Hybrid_Gated_QueryFusion", 
        "Hybrid_KAN_Gated_FeatureFusion", 
        "Hybrid_FiLM_Ablation",
        "Hybrid_Predict_Fusion",
        "Hybrid_NoHQ_Predict_Fusion",
        "htmformer",
        "MLP",
        "TransformerFixed",
        "MMK",
        "FiLM"
    ]
    
    print("\nSelect a model to run:")
    for i, m in enumerate(models):
        print(f"  {i+1}. {m}")
        
    print("\nSelect a model mode:")
    print("  y: Run ALL models (with option to exclude specific ones)")
    print("  n: Select a SINGLE model")
    
    mode_choice = input("\nRun all models? (y/n): ").strip().lower()
    
    run_models = []
    
    if mode_choice == 'y':
        # Batch Mode
        print("\n--- Batch Run Mode ---")
        print("Enter the numbers of models to EXCLUDE (space-separated, e.g., '1 3').")
        print("Leave blank to run ALL models.")
        
        exclude_input = input("Exclude IDs: ").strip()
        exclude_indices = set()
        if exclude_input:
            try:
                for x in exclude_input.split():
                    exclude_indices.add(int(x) - 1)
            except ValueError:
                print("⚠ Invalid input detected. Ignoring invalid numbers.")
        
        for i, m in enumerate(models):
            if i not in exclude_indices:
                run_models.append(m)
        
        if not run_models:
            print("❌ All models excluded! Exiting.")
            return

        # Resume Logic
        print(f"\n✅ Models to be run ({len(run_models)}):")
        for i, m in enumerate(run_models):
            print(f"  {i+1}. {m}")
            
        resume_choice = input("\nResume from a specific model in this list? (y/n): ").strip().lower()
        if resume_choice == 'y':
            try:
                start_num = int(input("Enter the number to start from (e.g., 5): "))
                if 1 <= start_num <= len(run_models):
                    print(f"⏩ Skipping first {start_num-1} models. Starting from {run_models[start_num-1]}...")
                    run_models = run_models[start_num-1:]
                else:
                    print("⚠ Invalid number. Running from start.")
            except ValueError:
                print("⚠ Invalid input. Running from start.")

        print("\nℹ️  Note: In batch mode, default settings will be used:")
        print("  - FiLM Mode: bidir")
        print("  - HQ: Enabled (True)")
        
        # Defaults for batch
        runner_config.FILM_MODE = "bidir"
        runner_config.USE_HQ = True

    else:
        # Single Mode (Original Logic)
        while True:
            try:
                choice = input("\nEnter number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    run_models = [models[idx]]
                    break
                else:
                    print("Invalid number.")
            except ValueError:
                print("Please enter a number.")
        
        selected_model = run_models[0]
        print(f"\n✅ Selected Model: {selected_model}")

        # Optional: Select FiLM Mode (if applicable) -> Only for Single Select
        if selected_model == "Hybrid_FiLM_Ablation":
            film_modes = ["bidir", "t2v", "v2t", "none"]
            print("\nSelect FiLM Mode (Modulation Direction):")
            print("  1. bidir (Bi-directional: Time <-> Variable) [Default]")
            print("  2. t2v   (Time -> Variable)")
            print("  3. v2t   (Variable -> Time)")
            print("  4. none  (No Modulation)")
            
            choice_f = input("\nEnter number (Enter for default): ").strip()
            if choice_f:
                try:
                    f_idx = int(choice_f) - 1
                    if 0 <= f_idx < len(film_modes):
                        runner_config.FILM_MODE = film_modes[f_idx]
                except ValueError:
                    pass
            print(f"✅ Selected FiLM Mode: {runner_config.FILM_MODE}")

        # Optional: Select HQ (BottleNeck vs Global)
        use_hq = True
        models_with_hq_option = ["Hybrid_FiLM_Ablation", "Hybrid_Gated_QueryFusion", "Hybrid_Predict_Fusion"]
        
        if selected_model in models_with_hq_option:
            print("\nUse Horizon Query (HQ)?")
            print("  - Yes (default): Use 24 distinct queries (Hour-specific attention)")
            print("    [Pros] Hour-specific patterns. [Cons] Bottleneck 72h -> 24 vectors.")
            print("  - No: Use 1 global query")
            print("    [Pros] Retains global context of 72h. [Cons] Less focus on individual future hours.")
            
            hq_choice = input("Use HQ? (y/n, default y): ").lower().strip()
            if hq_choice == 'n':
                runner_config.USE_HQ = False
            print(f"✅ HQ Enabled: {runner_config.USE_HQ}")

    # Common Settings Selection (Season)


    # Update Global Configs that are shared
    runner_config.SEASON = selected_season
    
    # Confirm
    print("\nConfiguration:")
    print(f"  - Models to run: {', '.join(run_models)}")
    print(f"  - Season: {selected_season}")
    
    if use_seasonal_files:
        if selected_season == "all":
            print(f"  - Data: submit_merged_with_residual.csv (in data/seasonal_split)")
        elif selected_season == "every":
             print(f"  - Data: [Dynamic] submit_merged_with_residual_{{season}}.csv")
        else:
             print(f"  - Data: submit_merged_with_residual_{selected_season}.csv (in data/seasonal_split)")
    else:
        if selected_season == "all":
             print(f"  - Data: demand_holiday_with_solar_mean.csv")
        elif selected_season == "every":
             print(f"  - Data: [Dynamic] demand_holiday_with_solar_mean_{{season}}.csv")
        else:
             print(f"  - Data: demand_holiday_with_solar_mean_{selected_season}.csv")
        
    print(f"  - Epochs: {runner_config.EPOCHS}")
    print(f"  - Horizon: {runner_config.HORIZON} hours")
    
    do_run = input("\nStart training? (y/n): ").lower()
    if do_run == 'y':
        for model_name in run_models:
            print(f"\n{'='*40}")
            print(f"🛳️  Processing Model: {model_name}")
            print(f"{'='*40}")
            runner_config.MODEL_NAME = model_name
            
            if selected_season == "every":
                run_seasons = ["spring", "summer", "autumn", "winter", "all"]
                for s in run_seasons:
                    print(f"\n🚀 Starting run for season: {s}")
                    runner_config.SEASON = s
                    
                    if use_seasonal_files:
                        # Residual Demand (Seasonal Split)
                        if s == "all":
                            fname = "submit_merged_with_residual.csv"
                        else:
                            fname = f"submit_merged_with_residual_{s}.csv"
                        runner_config.DATA_FILE = str(Path(BASE_DIR) / "data/seasonal_split" / fname)
                    else:
                        # Power Demand (Seasonal Files)
                        if s == "all":
                            fname = "demand_holiday_with_solar_mean.csv"
                        else:
                            fname = f"demand_holiday_with_solar_mean_{s}.csv"
                        runner_config.DATA_FILE = str(Path(BASE_DIR) / "data" / fname)
                        
                    print(f"📂 Using Seasonal File: {runner_config.DATA_FILE}")
                    
                    run_pipeline()
            else:
                if use_seasonal_files:
                    if selected_season == "all":
                        fname = "submit_merged_with_residual.csv"
                    else:
                        fname = f"submit_merged_with_residual_{selected_season}.csv"
                    runner_config.DATA_FILE = str(Path(BASE_DIR) / "data/seasonal_split" / fname)
                else:
                    if selected_season == "all":
                        fname = "demand_holiday_with_solar_mean.csv"
                    else:
                        fname = f"demand_holiday_with_solar_mean_{selected_season}.csv"
                    runner_config.DATA_FILE = str(Path(BASE_DIR) / "data" / fname)

                print(f"📂 Using Seasonal File: {runner_config.DATA_FILE}")
 
                run_pipeline()
                
        print("\n🎉🎉 All requested models completed! 🎉🎉")
    else:
        print("Cancelled.")

if __name__ == "__main__":
    interactive_main()
    
