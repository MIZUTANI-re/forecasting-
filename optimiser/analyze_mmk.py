
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import math

def plot_kan_experts(pth_path, output_dir, feature_names, seq_len=72, horizon=24, n_experts=4):
    """Plots the learned spline functions for all experts across all features."""
    from optimiser.models import MMKModel, MMKLayer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with standard parameters
    exog_dim = len(feature_names) - 1
    model = MMKModel(seq_len=seq_len, exog_dim=exog_dim, horizon=horizon, n_experts=n_experts, num_layers=1)
    
    try:
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.to(device)
        model.eval()
        
        output_dir = Path(output_dir) / "experts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        grid_range = [-1, 1]
        x_plot = torch.linspace(grid_range[0], grid_range[1], 100).to(device)
        
        for i, feat_name in enumerate(feature_names):
            branch = model.branches[i]
            # Handle num_layers=1 or >1
            mmk_layer = branch if isinstance(branch, MMKLayer) else branch.layers[0]
            
            plt.figure(figsize=(10, 6))
            for exp_idx, expert in enumerate(mmk_layer.experts):
                with torch.no_grad():
                    # Synthetic input (B, L)
                    # We vary only the most recent timestep for visualization
                    syn_in = torch.zeros(100, seq_len).to(device)
                    syn_in[:, -1] = x_plot
                    y_plot = expert(syn_in)[:, 0].cpu().numpy() # First horizon step
                
                plt.plot(x_plot.cpu().numpy(), y_plot, label=f"Expert {exp_idx}")
            
            plt.title(f"Expert Activation Functions: {feat_name}")
            plt.xlabel("Input Value (Normalized)")
            plt.ylabel("Prediction Contribution")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f"expert_{feat_name}.png")
            plt.close()
            
        print(f"🎨 Expert function gallery generated in {output_dir}")
    except Exception as e:
        print(f"⚠️ Could not generate expert plots: {e}")

def analyze_mmk_contributions(npz_path, output_dir, pth_path=None):
    try:
        data = np.load(npz_path, allow_pickle=True)
        # Check available keys
        print(f"📂 Loaded internals with keys: {list(data.keys())}")
        
        branch_outputs = data['branch_outputs'] # (Samples, F, Horizon)
        feature_names = list(data['feature_names'])
        
        # Add power_demand_past if missing
        if branch_outputs.shape[1] == len(feature_names) + 1:
            feature_names = ["power_demand_past"] + feature_names
        feature_names = np.array(feature_names)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Variable Importance (Mean Absolute Contribution)
        # sum across horizon or pick one
        horizon_idx = 0
        importance = np.mean(np.abs(branch_outputs[:, :, horizon_idx]), axis=0)
        
        plt.figure(figsize=(10, 8))
        sorted_idx = np.argsort(importance)
        plt.barh(feature_names[sorted_idx], importance[sorted_idx], color='skyblue')
        plt.title("MMK Global Feature Importance (Additive Decomposition)")
        plt.xlabel("Mean Abs Contribution (Scaled)")
        plt.tight_layout()
        plt.savefig(output_dir / "mmk_feature_importance.png")
        
        # 2. Decompose a specific sample (e.g., peak demand or first sample)
        sample_idx = 0
        contributions = branch_outputs[sample_idx, :, horizon_idx]
        
        plt.figure(figsize=(12, 6))
        plt.bar(feature_names, contributions, color='coral')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"MMK Contribution Breakdown for Sample {sample_idx}")
        plt.ylabel("Contribution to Prediction")
        plt.tight_layout()
        plt.savefig(output_dir / f"mmk_sample_{sample_idx}_breakdown.png")
        
        # 3. Routing Probabilities if available
        if 'routing_probs' in data:
            routing_probs = data['routing_probs'] # (Samples, F, 1, n_exp)
            avg_routing = np.mean(routing_probs[:, :, 0, :], axis=0)
            
            plt.figure(figsize=(12, 10))
            import seaborn as sns
            sns.heatmap(avg_routing, xticklabels=[f"Expert {i}" for i in range(avg_routing.shape[1])],
                       yticklabels=feature_names, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title("Expert Usage Map (Which KAN handles which variable?)")
            plt.tight_layout()
            plt.savefig(output_dir / "mmk_expert_routing_map.png")
            
        # 4. Specific Analysis: Weekday Influence
        # Identify weekday columns
        weekday_cols = [i for i, name in enumerate(feature_names) if "weekday" in name]
        if weekday_cols:
            # Average contribution of each weekday when it is 'active' (input=1)
            # Since these are one-hot, the contribution is roughly 
            # the bias/offset introduced by that day.
            weekday_labels = [feature_names[i] for i in weekday_cols]
            weekday_impacts = [np.mean(branch_outputs[:, i, horizon_idx]) for i in weekday_cols]
            
            plt.figure(figsize=(10, 6))
            plt.bar(weekday_labels, weekday_impacts, color='mediumpurple')
            plt.axhline(0, color='black', linewidth=0.8)
            plt.title("Net Impact of Each Weekday on Power Demand Prediction")
            plt.ylabel("Baseline Offset (Scaled)")
            plt.tight_layout()
            plt.savefig(output_dir / "mmk_weekday_influence.png")
            
        # 5. Expert Activation Functions if weights are provided
        if pth_path and Path(pth_path).exists():
            plot_kan_experts(pth_path, output_dir, feature_names)
            
        print(f"✨ MMK Interpretability plots generated in {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during MMK analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    npz_path = r"c:\Users\2213144\practice\result_csv\MMK\MMK_spring_separated_internals.npz"
    output_dir = r"c:\Users\2213144\practice\picture\MMK_Analysis"
    if Path(npz_path).exists():
        analyze_mmk_contributions(npz_path, output_dir)
    else:
        print(f"❌ Could not find {npz_path}")
