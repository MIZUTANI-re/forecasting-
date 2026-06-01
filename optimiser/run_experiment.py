
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Add project root to sys.path to ensure modules are found when running this script directly
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from optimiser.data_loader import DataLoaderFactory
from optimiser.models import get_model
from optimiser.evaluation import Evaluator
from optimiser.hyperparameter_search import HyperparameterSampler

# ============================================================
# ✅ User Configuration (Edit here)
# ============================================================

# Model Selection
MODEL_NAME = "FiLM"

# Operation Mode: "fixed" or "optimize"
MODE = "fixed" # Switch this based on user request
OPTIMIZATION_TRIALS = 10 # Number of trials for optimization
USE_OPTIMIZED_PARAMS = False # Flag to use best params from reserch

# Data: "every" to run all seasons
SEASON = "spring"
DATA_FILE = r"C:\Users\2213144\practice\data\seasonal_split\submit_merged_with_residual.csv" 
# Target Variable
TARGET_COL = "power_demand"
# Note: For specific seasons, point to the correct file

# Run Settings
# FiLM Configuration (Modulation Direction)
# "t2v"   : Time -> Variable (Time modulates Variable)
# "v2t"   : Variable -> Time (Variable modulates Time)
# "bidir" : Bidirectional (Both modulate each other)
# "none"  : No modulation (Standard Fusion)
FILM_MODE = "bidir"
USE_HQ = True
SEQ_LEN = 72
HORIZON = 24
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

# IEEJ Comparison Mode
IEEJ_COMPARISON = False 
IEEJ_DATES = {
    "train_end": "2020-04-30",
    "test_start": "2021-01-01",
    "test_end": "2021-12-31"
}

# COVID-19 Exclusion Mode
EXCLUDE_COVID = False
COVID_RANGE = [
    {"start": "2020-02-01", "end": "2022-12-31"}
]

# MMK Lightweight Mode (Consolidates categorical features and reduces experts)
USE_LIGHTWEIGHT_MMK = True

# Model Hyperparameters
HP = {
    "d_model": 128,
    "nhead": 4,      # For HTMformer, this is nhead_var
    "num_layers": 2,
    "hidden_dim": 256,
    "dropout": 0.1,
    
    # Model Specific
    "nhead_time": 4, # HTMformer only
    "cnn_kernel": 3, # Hybrid/KAN only
    "cnn_mode": "linear", # "linear", "cnn1", "cnn3"
    "modes": 32, # FiLM / FFT modes
}

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = BASE_DIR / "result_csv"
MODEL_DIR = BASE_DIR / "learning_saver"
FIG_DIR = BASE_DIR / "picture"
SUMMARY_PATH = RESULT_DIR / "experiment_summary.csv"

# ============================================================
# Main Runner
# ============================================================

def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("\n🎉 Done! Check 'result_csv' and 'runs' folders.")

def train_eval_model(current_hp, info, train_loader, val_loader, test_loader, device, dataset_mode, current_model_dir, current_result_dir, current_fig_dir, current_summary_path, trial_id=0):
    
    # 3. Initialize Model with current_hp
    model_config = current_hp.copy()
    model_config.update({
        "seq_len": SEQ_LEN,
        "pred_len": HORIZON,
        "in_dim": info["in_dim"] if dataset_mode == "unified" else info["exog_dim"],
        # HTMformer specific adjustment
        "nhead_var": current_hp.get("nhead", 4),
        "film_mode": current_hp.get("film_mode", FILM_MODE),
        "use_hq": USE_HQ
    })
    
    # Handle Lightweight MMK config (Heterogeneous Experts)
    if MODEL_NAME == "MMK" and USE_LIGHTWEIGHT_MMK:
        # We need n_experts for each variable: [target_past, exog1, exog2, ...]
        n_experts_list = []
        base_n = current_hp.get("n_experts_base", 4)
        
        # 1. Past demand (Complex)
        n_experts_list.append(base_n) 
        
        # 2. Exogenous features
        exog_feats = info.get("feature_names", [])[1:] if "feature_names" in info else []
        for feat in exog_feats:
            if any(k in feat for k in ["weekday", "month", "holiday", "restday"]):
                n_experts_list.append(base_n) # Hybrid experiment: 4 experts for cyclic features
            else:
                n_experts_list.append(base_n) # Continuous (temp, precip, etc.)
        
        if len(n_experts_list) == (info.get("exog_dim", 0) + 1):
             model_config["n_experts"] = n_experts_list
             print(f"📉 MMK Lightweight Mode: Reduced expert counts per feature: {n_experts_list}")
    
    model = get_model(MODEL_NAME, model_config).to(device)
    
    # --- Count Parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    # print(f"📊 Model Complexity: {total_params:,} parameters")
    
    # Use AdamW as requested for optimization
    lr = current_hp.get("learning_rate", LEARNING_RATE)
    optimizer = optim.AdamW(model.parameters(), lr=lr) 
    criterion = nn.MSELoss()

    # 4. Train Loop
    best_val_loss = float("inf")
    bad_counter = 0
    save_run_name = f"{MODEL_NAME}_{SEASON}_{dataset_mode}_trial{trial_id}"
    save_path = current_model_dir / f"{save_run_name}.pth"
    current_model_dir.mkdir(parents=True, exist_ok=True)

    # print(f"\n🔄 Training Trial {trial_id}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            if dataset_mode == "unified":
                x, y = batch
                x, y = x.to(device), y.to(device)
                pred = model(x)
                idx = info["target_idx"]
                if idx != -1 and pred.dim() == 3: pred = pred[:, :, idx]
            else:
                x_ex, y_p, y_f = batch
                x_ex, y_p, y_f = x_ex.to(device), y_p.to(device), y_f.to(device)
                pred = model(x_ex, y_p)
                y = y_f
                
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if dataset_mode == "unified":
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    idx = info["target_idx"]
                    if idx != -1 and pred.dim() == 3: pred = pred[:, :, idx]
                else:
                    x_ex, y_p, y_f = batch
                    x_ex, y_p, y_f = x_ex.to(device), y_p.to(device), y_f.to(device)
                    pred = model(x_ex, y_p)
                    y = y_f
                val_loss += criterion(pred, y).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            bad_counter += 1
            if bad_counter >= 20: # Fixed Patience 20
                # print(f"⏹ Early stopping at epoch {epoch}")
                break

    # 5. Evaluate Best Model of Trial
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    preds_list = []
    trues_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            if dataset_mode == "unified":
                x, y = batch
                x, y = x.to(device), y.to(device)
                pred = model(x)
                idx = info["target_idx"]
                if idx != -1 and pred.dim() == 3: pred = pred[:, :, idx]
            else:
                x_ex, y_p, y_f = batch
                x_ex, y_p, y_f = x_ex.to(device), y_p.to(device), y_f.to(device)
                pred = model(x_ex, y_p)
                y = y_f
            
            preds_list.append(pred.cpu().numpy())
            trues_list.append(y.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)

    scaler = info["scaler"]
    preds_inv = scaler.inverse_transform_target(preds)
    trues_inv = scaler.inverse_transform_target(trues)

    # Calculate Metrics
    # from optimiser.evaluation import MetricCalculator
    # metrics = MetricCalculator.calculate_metrics(trues_inv, preds_inv)
    
    yt_flat = trues_inv.reshape(-1)
    yp_flat = preds_inv.reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(yt_flat, yp_flat)))
    mape = float(mean_absolute_percentage_error(yt_flat, yp_flat) * 100.0)
    metrics = {"rmse": rmse, "mape": mape}
    
    return metrics, best_val_loss, save_path, preds_inv, trues_inv, total_params

def run_experiment_manager():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Run | Model: {MODEL_NAME} | Mode: {MODE} | Device: {device}")
    
    # 0. Setup Paths based on Target and Horizon
    if HORIZON == 72:
        base_name_res = "result_72_predict"
        base_name_mod = "learning_saver_72"
        base_name_fig = "picture_72"
    else:
        base_name_res = "result_csv"
        base_name_mod = "learning_saver"
        base_name_fig = "picture"

    if TARGET_COL == "residual_demand":
        base_name_res += "_residual"
        base_name_mod += "_residual"
        base_name_fig += "_residual"

    current_result_dir = BASE_DIR / base_name_res
    current_model_dir = BASE_DIR / base_name_mod
    current_fig_dir = BASE_DIR / base_name_fig
    
    if MODEL_NAME == "MMK":
        current_summary_path = current_result_dir / "experiment_summary_mmk_hybrid.csv"
    else:
        current_summary_path = current_result_dir / "experiment_summary.csv"

    # 1. Dataset Mode
    dataset_mode = "unified"
    if MODEL_NAME in ["MMK", "grid_tst", "Hybrid_Gated_QueryFusion", "Hybrid_KAN_Gated_FeatureFusion", "Hybrid_Gated_FeatureFusion", "Hybrid_FiLM_Ablation", "Hybrid_Predict_Fusion", "Hybrid_NoHQ_Predict_Fusion"]:
        dataset_mode = "separated"
    
    print(f"📊 Dataset Mode: {dataset_mode}")

    # 2. Load Data (Common for all trials)
    loader_factory = DataLoaderFactory(
        seq_len=SEQ_LEN, 
        horizon=HORIZON,
        target_col=TARGET_COL,
        use_onehot=(not USE_LIGHTWEIGHT_MMK) if MODEL_NAME == "MMK" else True,
        use_cyclic=(MODEL_NAME == "MMK")
    )
    
    # We might need to adjust batch size dynamically if optimized, 
    # but for Data Loading we'll use a default or the first sampled one.
    # Actually, batch_size is a loading param. If we optimize batch_size, we need to reload data or split manually.
    # To keep it simple, let's fix batch_size for loading or reload if needed.
    # The user asked to optimize "everything else", usually batch_size is influential.
    # Let's assume loading happens once with default BATCH_SIZE to save time, or we reload inside loop if batch_size changes.
    # For efficiency, let's keep BATCH_SIZE fixed to 64 for data loading, or only reload if absolutely necessary.
    # Actually, HyperparameterSampler includes batch_size. We MUST reload data if batch size changes.
    
    # We will call load_data inside the optimization loop to handle batch_size changes.

    
    sampler = HyperparameterSampler()
    
    if MODE == "optimize":
        import optuna
        print(f"🔎 Optimization Mode: Running {OPTIMIZATION_TRIALS} trials via Optuna")
        
        def objective(trial):
            params = sampler.optuna_sample_params(trial, MODEL_NAME)
            current_bs = params.get("batch_size", BATCH_SIZE)
            
            print(f"--- Optuna Trial {trial.number}/{OPTIMIZATION_TRIALS} | BS: {current_bs} | LR: {params.get('learning_rate')} ---")
            
            # Load Data
            train_loader, val_loader, test_loader, info = loader_factory.load_data(
                file_path=DATA_FILE,
                mode=dataset_mode,
                batch_size=current_bs,
                season=SEASON,
                custom_dates=IEEJ_DATES if IEEJ_COMPARISON else None,
                exclude_dates=COVID_RANGE if EXCLUDE_COVID else None
            )
            
            # Train and Eval
            metrics, val_loss, model_path, preds, trues, params_count = train_eval_model(
                params, info, train_loader, val_loader, test_loader, device, dataset_mode,
                current_model_dir, current_result_dir, current_fig_dir, current_summary_path, trial_id=trial.number
            )
            
            # Store artifacts for retrieval
            trial.set_user_attr("params", params)
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("path", model_path)
            trial.set_user_attr("preds", preds)
            trial.set_user_attr("trues", trues)
            trial.set_user_attr("count", params_count)
            
            print(f"    RMSE: {metrics['rmse']:.4f} | MAPE: {metrics['mape']:.4f}% | Val Loss: {val_loss:.6f}")
            return val_loss

        # Run Study
        optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce optuna noise
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=OPTIMIZATION_TRIALS)
        
        # Select Best
        best_optuna_trial = study.best_trial
        best_trial = {
            "trial": best_optuna_trial.number,
            "params": best_optuna_trial.user_attrs["params"],
            "metrics": best_optuna_trial.user_attrs["metrics"],
            "val_loss": best_optuna_trial.value,
            "path": best_optuna_trial.user_attrs["path"],
            "preds": best_optuna_trial.user_attrs["preds"],
            "trues": best_optuna_trial.user_attrs["trues"],
            "count": best_optuna_trial.user_attrs["count"]
        }
        
        print("\n🏆 OPTIMIZATION COMPLETE 🏆")
        print(f"Best Trial: {best_trial['trial']} (Val Loss: {best_trial['val_loss']:.6f})")
        print(f"Params: {best_trial['params']}")
        
        # Save Best Results Final
        run_name = f"{MODEL_NAME}_{SEASON}_{dataset_mode}_optimized"
        
        # We use the preds/trues from the best trial
        evaluator = Evaluator(current_result_dir / MODEL_NAME, current_fig_dir / MODEL_NAME)
        evaluator.result_dir.mkdir(parents=True, exist_ok=True)
        evaluator.fig_dir.mkdir(parents=True, exist_ok=True)
        
        config_dump = {
            "model": MODEL_NAME,
            "season": SEASON, 
            "target": TARGET_COL,
            "rmse": best_trial["metrics"]["rmse"],
            "mape": best_trial["metrics"]["mape"],
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "optimized": True,
            "params_count": best_trial["count"],
            **best_trial["params"]
        }
        
        csv_path = evaluator.save_prediction_csv(run_name, best_trial["preds"], best_trial["trues"], info, config_dump)
        evaluator.plot_results(run_name, csv_path)
        evaluator.append_summary(current_summary_path, config_dump)
        
    else:
        # Fixed Mode
        if USE_OPTIMIZED_PARAMS:
            print(f"Using OPTIMIZED parameters for {MODEL_NAME} ({SEASON})")
            params = sampler.get_optimized_params(MODEL_NAME, SEASON)
        else:
            print(f"Using DEFAULT fixed parameters for {MODEL_NAME} ({SEASON})")
            params = sampler.get_fixed_params()
        
        train_loader, val_loader, test_loader, info = loader_factory.load_data(
            file_path=DATA_FILE,
            mode=dataset_mode,
            batch_size=params.get("batch_size", BATCH_SIZE),
            season=SEASON,
            custom_dates=IEEJ_DATES if IEEJ_COMPARISON else None,
            exclude_dates=COVID_RANGE if EXCLUDE_COVID else None
        )
        
        metrics, val, path, preds, trues, count = train_eval_model(
            params, info, train_loader, val_loader, test_loader, device, dataset_mode,
            current_model_dir, current_result_dir, current_fig_dir, current_summary_path, trial_id=0
        )
        
        print(f"✅ Results: RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.4f}%")
        
        # Save
        run_name = f"{MODEL_NAME}_{SEASON}_{dataset_mode}"
        evaluator = Evaluator(current_result_dir / MODEL_NAME, current_fig_dir / MODEL_NAME)
        evaluator.result_dir.mkdir(parents=True, exist_ok=True)
        evaluator.fig_dir.mkdir(parents=True, exist_ok=True)
        
        config_dump = {
            "model": MODEL_NAME,
            "season": SEASON, 
            "target": TARGET_COL,
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "seq_len": SEQ_LEN,
            "pred_len": HORIZON,
            "optimized": False,
            "params_count": count,
            **params
        }
        csv_path = evaluator.save_prediction_csv(run_name, preds, trues, info, config_dump)
        evaluator.plot_results(run_name, csv_path)
        evaluator.append_summary(current_summary_path, config_dump)

def main():
    global SEASON, DATA_FILE
    
    initial_season = SEASON
    if initial_season == "every":
        target_seasons = ["spring", "summer", "autumn", "winter", "all"]
    else:
        target_seasons = [initial_season]
        
    for current_s in target_seasons:
        SEASON = current_s
        print(f"\n{'='*60}")
        print(f"🌟 Starting Season: {SEASON} (Target: {TARGET_COL})")
        print(f"{'='*60}")
        
        if TARGET_COL == "residual_demand":
            fname = "submit_merged_with_residual.csv" if SEASON == "all" else f"submit_merged_with_residual_{SEASON}.csv"
            DATA_FILE = str(BASE_DIR / "data/seasonal_split" / fname)
        else: 
            fname = "demand_holiday_with_solar_mean.csv" if SEASON == "all" else f"demand_holiday_with_solar_mean_{SEASON}.csv"
            DATA_FILE = str(BASE_DIR / "data" / fname)
            
        print(f"📂 Using Data: {DATA_FILE}")

        run_experiment_manager()

if __name__ == "__main__":
    main()
