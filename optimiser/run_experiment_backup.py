
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path to ensure modules are found when running this script directly
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from optimiser.data_loader import DataLoaderFactory
from optimiser.models import get_model
from optimiser.evaluation import Evaluator

# ============================================================
# ✅ User Configuration (Edit here)
# ============================================================

# Model Selection
MODEL_NAME = "MMK"

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
EPOCHS = 50
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
EXCLUDE_COVID = True
COVID_RANGE = [
    {"start": "2020-02-01", "end": "2022-12-31"}
]

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
    "n_experts": 4,  # MMK only
    "grid_size": 3,  # MMK only
    "num_layers": 1, # MMK only
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

def run_single_experiment():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Run | Model: {MODEL_NAME} | Device: {device}")
    
    # 0. Setup Paths based on Target and Horizon
    # Determine base suffix based on Horizon
    if HORIZON == 72:
        base_name_res = "result_72_predict"
        base_name_mod = "learning_saver_72"
        base_name_fig = "picture_72"
    else:
        # Use top-level directory names
        base_name_res = RESULT_DIR.name
        base_name_mod = MODEL_DIR.name
        base_name_fig = FIG_DIR.name

    # Add residual suffix if needed
    if TARGET_COL == "residual_demand":
        base_name_res += "_residual"
        base_name_mod += "_residual"
        base_name_fig += "_residual"

    current_result_dir = RESULT_DIR.parent / base_name_res
    current_model_dir = MODEL_DIR.parent / base_name_mod
    current_fig_dir = FIG_DIR.parent / base_name_fig
    
    # Ensure they exist
    current_result_dir.mkdir(parents=True, exist_ok=True)
    current_model_dir.mkdir(parents=True, exist_ok=True)
    current_fig_dir.mkdir(parents=True, exist_ok=True)
    
    current_summary_path = current_result_dir / "experiment_summary.csv"

    # 1. Determine Data Mode based on Model
    # Some models need separated input (X_exog, y_past) vs Unified (X)
    dataset_mode = "unified"
    if MODEL_NAME in ["MMK", "grid_tst", "Hybrid_Gated_QueryFusion", "Hybrid_KAN_Gated_FeatureFusion", "Hybrid_Gated_FeatureFusion", "Hybrid_FiLM_Ablation", "Hybrid_Predict_Fusion", "Hybrid_NoHQ_Predict_Fusion"]:
        dataset_mode = "separated"
    
    print(f"📊 Dataset Mode: {dataset_mode}")

    # 2. Load Data
    loader_factory = DataLoaderFactory(
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        strict_split=True,
        sequence_mode="warm",
        target_col=TARGET_COL
    )
    
    train_loader, val_loader, test_loader, info = loader_factory.load_data(
        file_path=DATA_FILE,
        mode=dataset_mode,
        batch_size=BATCH_SIZE,
        season=SEASON,  # Pass global SEASON variable
        custom_dates=IEEJ_DATES if IEEJ_COMPARISON else None,
        exclude_dates=COVID_RANGE if EXCLUDE_COVID else None
    )
    
    print(f"📦 Data Loaded. Train batches: {len(train_loader)}")

    # 3. Initialize Model
    model_config = HP.copy()
    model_config.update({
        "seq_len": SEQ_LEN,
        "pred_len": HORIZON,
        "in_dim": info["in_dim"] if dataset_mode == "unified" else info["exog_dim"],
        # HTMformer specific adjustment
        "nhead_var": HP["nhead"],
        "film_mode": FILM_MODE,
        "use_hq": USE_HQ,
        **HP 
    })
    
    model = get_model(MODEL_NAME, model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 4. Train Loop
    best_val = float("inf")
    patience = 10
    bad_counter = 0
    # 4. Train Loop
    best_val = float("inf")
    patience = 10
    bad_counter = 0
    save_path = current_model_dir / f"{MODEL_NAME}_{SEASON}_{dataset_mode}.pth"
    current_model_dir.mkdir(parents=True, exist_ok=True)

    print("\n🔄 Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Unpack batch based on mode
            if dataset_mode == "unified":
                x, y = batch
                x, y = x.to(device), y.to(device)
                pred = model(x)
                # Slice target if output is multivariate (B, H, D)
                idx = info["target_idx"]
                if idx != -1 and pred.dim() == 3:
                    pred = pred[:, :, idx]
                else:
                    # Already univariate (B, H) or fallback
                    pass
            else:
                x_ex, y_p, y_f = batch
                x_ex, y_p, y_f = x_ex.to(device), y_p.to(device), y_f.to(device)
                pred = model(x_ex, y_p)
                y = y_f # Target for loss
                
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
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
                    if idx != -1 and pred.dim() == 3:
                        pred = pred[:, :, idx]
                else:
                    x_ex, y_p, y_f = batch
                    x_ex, y_p, y_f = x_ex.to(device), y_p.to(device), y_f.to(device)
                    pred = model(x_ex, y_p)
                    y = y_f

                val_loss += criterion(pred, y).item()
        
        val_loss /= len(val_loader)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Val Loss: {val_loss:.6f}")
            
        if val_loss < best_val:
            best_val = val_loss
            bad_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            bad_counter += 1
            if bad_counter >= patience:
                print(f"⏹ Early stopping at epoch {epoch}")
                break

    # 5. Evaluate
    print("\n📝 Evaluating...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    preds_list = []
    trues_list = []
    internal_states = {} # To store gate values etc.
    
    with torch.no_grad():
        for batch in test_loader:
            # Check if model supports returning internals
            try:
                if dataset_mode == "unified":
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    # Try calling with return_internals
                    try:
                        p, internals = model(x, return_internals=True)
                        if isinstance(internals, dict):
                            for k, v in internals.items():
                                if k not in internal_states:
                                    internal_states[k] = []
                                internal_states[k].append(v.cpu().numpy())
                    except TypeError:
                        p = model(x)
                        
                    idx = info["target_idx"]
                    if idx != -1 and p.dim() == 3:
                        p = p[:, :, idx]
                else:
                    x_ex, y_p, y_f = batch
                    x_ex, y_p, y_f = x_ex.to(device), y_p.to(device), y_f.to(device)
                    # Try calling with return_internals
                    try:
                        p, internals = model(x_ex, y_p, return_internals=True)
                        if isinstance(internals, dict):
                            for k, v in internals.items():
                                if k not in internal_states:
                                    internal_states[k] = []
                                internal_states[k].append(v.cpu().numpy())
                    except TypeError:
                        p = model(x_ex, y_p)

                    y = y_f
            except Exception as e:
                # Fallback
                 if dataset_mode == "unified":
                     x, y = batch
                     x, y = x.to(device), y.to(device)
                     p = model(x)
                     idx = info["target_idx"]
                     if idx != -1 and p.dim() == 3:
                         p = p[:, :, idx]
                 else:
                     x_ex, y_p, y_f = batch
                     x_ex, y_p, y_f = x_ex.to(device), y_p.to(device), y_f.to(device)
                     p = model(x_ex, y_p)
                     y = y_f
            
            preds_list.append(p.cpu().numpy())
            trues_list.append(y.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)

    # Inverse Transform
    scaler = info["scaler"]
    # Handle inverse based on shape (target is 1D per horizon step usually handled by scaler helper)
    preds_inv = []
    trues_inv = []
    # Currently scaler support basic inverse. Ideally we inverse per-element or batch.
    # ScalerWrapper.inverse_transform_target handles this.
    preds_inv = scaler.inverse_transform_target(preds)
    trues_inv = scaler.inverse_transform_target(trues)

    # 6. Save Results
    # Organize by Model Name
    model_result_dir = current_result_dir / MODEL_NAME
    model_fig_dir = current_fig_dir / MODEL_NAME
    evaluator = Evaluator(model_result_dir, model_fig_dir)

    # Save internals if any
    if internal_states:
        # Concatenate lists
        final_internals = {k: np.concatenate(v, axis=0) for k, v in internal_states.items()}
        # Add feature names if available for better plotting
        if "feature_names" in info:
            final_internals["feature_names"] = np.array(info["feature_names"])
            
        np.savez_compressed(model_result_dir / f"{MODEL_NAME}_{SEASON}_{dataset_mode}_internals.npz", **final_internals)
        print(f"📦 Internal states saved to {model_result_dir}")
    metrics = evaluator.calculate_metrics(trues_inv, preds_inv)
    
    print(f"✅ Results: RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.4f}%")
    
    run_name = f"{MODEL_NAME}_{SEASON}_{dataset_mode}"
    
    config_dump = {
        "model": MODEL_NAME,
        "season": SEASON, 
        "target": TARGET_COL,
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "pred_len": HORIZON,
        "exclude_covid": EXCLUDE_COVID,
        "film_mode": FILM_MODE,
        **HP
    }
    
    csv_path = evaluator.save_prediction_csv(
        run_name=run_name,
        preds=preds_inv,
        trues=trues_inv,
        info=info,
        config=config_dump
    )
    
    evaluator.plot_results(run_name, csv_path)
    evaluator.append_summary(current_summary_path, config_dump)
    
    print("\n🎉 Done! Check 'result_csv' and 'runs' folders.")

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
        
        # Dynamically set DATA_FILE based on target and season
        if TARGET_COL == "residual_demand":
            fname = "submit_merged_with_residual.csv" if SEASON == "all" else f"submit_merged_with_residual_{SEASON}.csv"
            DATA_FILE = str(BASE_DIR / "data/seasonal_split" / fname)
        else: # power_demand
            fname = "demand_holiday_with_solar_mean.csv" if SEASON == "all" else f"demand_holiday_with_solar_mean_{SEASON}.csv"
            DATA_FILE = str(BASE_DIR / "data" / fname)
            
        print(f"📂 Using Data: {DATA_FILE}")

        run_single_experiment()

if __name__ == "__main__":
    main()
