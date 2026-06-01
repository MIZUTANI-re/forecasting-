import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from visualize_models_influence folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root


from optimiser.models import get_model, MMKModel

def visualize_kan_functions():
    # Configuration (Must match the trained Hybrid model)
    # We need to reconstruct the model structure to load weights.
    # Hybrid settings:
    # - Cyclic features -> 10 features
    # - 4 experts for ALL features
    
    # 1. Define Model Config
    # Input Dim = 10 (Target + 3 Cont + 2 Week + 2 Month + 2 Bool)
    # Exog Dim = 9
    dim_in = 10 
    n_experts_list = [4] * dim_in # Hybrid configuration
    
    config = {
        "seq_len": 72,
        "pred_len": 24, # Horizon
        "in_dim": dim_in - 1, # Exog dim
        "d_model": 128,
        "n_experts": n_experts_list,
        "grid_size": 3,
        "num_layers": 1
    }
    
    device = torch.device("cpu")
    model = MMKModel(
        seq_len=config["seq_len"],
        exog_dim=config["in_dim"],
        horizon=config["pred_len"],
        d_model=config["d_model"],
        n_experts=config["n_experts"],
        grid_size=config["grid_size"],
        num_layers=config["num_layers"]
    ).to(device)
    
    # 2. Load Weights 
    # Path: learning_saver/MMK_spring_separated.pth
    # Wait, in run_experiment.py, save path is:
    # current_model_dir / f"{MODEL_NAME}_{SEASON}_{dataset_mode}.pth"
    # -> learning_saver/MMK_spring_separated.pth
    # Let's verify this path exists
    
    model_path = Path(r"c:\Users\2213144\practice\learning_saver\MMK_all_separated.pth")
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return

    print(f"Loading model from: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location=device)
        print("Debugging State Dict Mismatch:")
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        
        print(f"Missing keys (in model but not ckpt): {len(missing)}")
        if missing:
            print(f"Sample missing: {list(missing)[:5]}")
            
        print(f"Unexpected keys (in ckpt but not model): {len(unexpected)}")
        if unexpected:
            print(f"Sample unexpected: {list(unexpected)[:5]}")

        model.load_state_dict(state_dict, strict=False) # Try loose loading
    except Exception as e:
        print(f"❌ Failed to load state dict: {e}")
        print("Note: If the saved model has different parameters (e.g. from Lightweight run), this will fail.")
        print("Please ensure the last run was indeed the Hybrid run.")
        return

    model.eval()
    
    # 3. Visualize Functions
    # We want to see how experts respond to inputs (Cyclic features range from -1 to 1)
    
    # Feature Names reconstruction
    # 0: Target (Power)
    # 1: Temp
    # 2: Precip
    # 3: Solar
    # 4: W_Sin
    # 5: W_Cos
    # 6: M_Sin
    # 7: M_Cos
    # 8: Holiday
    # 9: Restday
    
    feature_map = {
        6: "Month Sin",
        7: "Month Cos",
        4: "Weekday Sin",
        5: "Weekday Cos",
        0: "Past Power (Normalized)"
    }
    
    save_dir = Path(r"c:\Users\2213144\practice\picture_analysis")
    save_dir.mkdir(exist_ok=True)
    
    # Data Collection
    results = {}
    sweep = np.linspace(-2, 2, 100)
    
    global_min = -0.5 # Default range or computed dynamically
    global_max = 0.6
    
    print("Computing responses for all features...")
    
    # Collect individual expert responses
    for feat_idx, feat_name in feature_map.items():
        if feat_idx < len(model.branches):
            branch = model.branches[feat_idx]
            
            expert_responses = []
            for expert in branch.experts:
                vals = []
                for v in sweep:
                    inp = torch.full((1, config["seq_len"]), v, device=device)
                    with torch.no_grad():
                        out = expert(inp)
                        vals.append(out.mean().item()) # Average over OUTPUT dimensions (horizon), not experts
                expert_responses.append(vals)
            
            # Store LIST of experts (not mean)
            results[feat_name] = expert_responses

    # --- 1. Save Individual Plots (One by One) ---
    print("\nSaving individual feature plots...")
    for feat_name, expert_lists in results.items():
        plt.figure(figsize=(10, 6))
        
        # Plot ALL experts for this feature
        for i, vals in enumerate(expert_lists):
            plt.plot(sweep, vals, label=f"Expert {i}", linewidth=2, alpha=0.8)
            
        plt.title(f"feature: {feat_name}\n(Individual Experts)", fontsize=14)
        plt.xlabel("Input (Norm)")
        plt.ylabel("Contribution")
        plt.ylim(global_min, global_max)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        safe_name = feat_name.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = save_dir / f"MMK_Func_{safe_name}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"  Saved: {out_path}")

    # --- 2. Static Plot (Summary) ---
    plt.figure(figsize=(10, 6))
    for feat_name, expert_lists in results.items():
        mean_vals = np.mean(expert_lists, axis=0)
        lw = 4 if "Power" in feat_name else 2
        alpha = 0.9 if "Power" in feat_name else 0.7
        color = 'black' if "Power" in feat_name else None
        plt.plot(sweep, mean_vals, label=feat_name, linewidth=lw, alpha=alpha, color=color)
            
    plt.title("MMK Additive Model: Summary of Feature Contributions\n(Mean of Experts)", fontsize=14)
    plt.xlabel("Input Feature Value (Normalized)")
    plt.ylabel("Output Contribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = save_dir / "MMK_OneGraph_Summary.png"
    plt.savefig(out_path)
    print(f"  Saved Summary Plot: {out_path}")
    plt.close()

    # --- 3. Interactive "Detailed Expert View" ---
    print("\n⚠️  Interactive Mode: Press any key/click on the plot window to advance.")
    
    try:
        import matplotlib
        if matplotlib.get_backend() == 'agg':
            print("    Skipping interactive mode (Backend is Agg)")
        else:
            # Interactive Loop
            for feat_name, expert_lists in results.items():
                plt.figure(figsize=(10, 6))
                for i, vals in enumerate(expert_lists):
                    plt.plot(sweep, vals, label=f"Expert {i}", linewidth=2)
                
                plt.title(f"feature: {feat_name} (Click to Next)", fontsize=14)
                plt.xlabel("Input")
                plt.ylabel("Contribution")
                plt.ylim(global_min, global_max)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                print(f"    Displaying '{feat_name}' detailed view... (Click to Next)")
                if plt.waitforbuttonpress(): 
                    pass
                plt.close()
                
    except Exception as e:
        print(f"    Interactive mode failed or skipped: {e}")

if __name__ == "__main__":
    visualize_kan_functions()
