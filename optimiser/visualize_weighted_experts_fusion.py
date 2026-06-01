
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from optimiser.models import MMKModel, MMKLayer

def plot_weighted_expert_fusion(pth_path, npz_path, output_dir, sample_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load internal data to get weights and feature names
    data = np.load(npz_path, allow_pickle=True)
    routing_probs = data['routing_probs'] # (Samples, F, L, n_exp)
    feature_names = list(data['feature_names'])
    
    # Check if power_demand_past is included
    if routing_probs.shape[1] == len(feature_names) + 1:
        feature_names = ["power_demand_past"] + feature_names
    
    # Initialize model
    exog_dim = len(feature_names) - 1
    model = MMKModel(seq_len=72, exog_dim=exog_dim, horizon=24, n_experts=4, num_layers=1)
    
    try:
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.to(device)
        model.eval()
        
        output_dir = Path(output_dir) / "weighted_fusion"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        x_plot = torch.linspace(-1, 1, 100).to(device)
        
        for feat_name in feature_names:
            feat_idx = feature_names.index(feat_name)
            weights = routing_probs[sample_idx, feat_idx, 0, :] # Weights for this sample
            
            branch = model.branches[feat_idx]
            mmk_layer = branch if isinstance(branch, MMKLayer) else branch.layers[0]
            
            plt.figure(figsize=(10, 6))
            
            # 1. Plot individual weighted experts
            total_y = np.zeros(100)
            colors = plt.cm.viridis(np.linspace(0, 0.8, 4))
            
            for i, expert in enumerate(mmk_layer.experts):
                with torch.no_grad():
                    syn_in = torch.zeros(100, 72).to(device)
                    syn_in[:, -1] = x_plot
                    y_raw = expert(syn_in)[:, 0].cpu().numpy()
                    
                    w = weights[i]
                    y_weighted = w * y_raw
                    total_y += y_weighted
                
                plt.plot(x_plot.cpu().numpy(), y_weighted, label=f'w{i+1}*f{i+1} ({w*100:.1f}%)', 
                         color=colors[i], alpha=0.6, linestyle='--')
            
            # 2. Plot the resulting Fused Formula
            plt.plot(x_plot.cpu().numpy(), total_y, label='RESULT (Total Formula)', 
                     color='red', linewidth=3)
            
            plt.title(f"Weighted Expert Fusion: {feat_name}\n"
                      f"Formula: y = {' + '.join([f'{weights[i]:.2f}*f{i+1}' for i in range(4)])}")
            plt.xlabel("Input Value (Normalized)")
            plt.ylabel("Contribution to Target")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = output_dir / f"fusion_{feat_name}.png"
            plt.savefig(save_path)
            plt.close()
            
        print(f"✨ Weighted fusion plots generated in {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pth = r'c:\Users\2213144\practice\result_csv\MMK\MMK_spring_separated.pth'
    npz = r'c:\Users\2213144\practice\result_csv\MMK\MMK_spring_separated_internals.npz'
    out = r'c:\Users\2213144\practice\picture\MMK_Analysis'
    plot_weighted_expert_fusion(pth, npz, out)
