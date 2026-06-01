import sys
from pathlib import Path
# Ensure project root is in sys.path so imports work from visualize_models_influence folder
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_heatmap(data, labels, title, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(data, annot=False, cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🖼️ iTransformer Heatmap saved: {save_path}")

def main():
    base_dir = Path(r"c:\Users\2213144\practice\result_csv_residual")
    output_dir = Path(r"c:\Users\2213144\practice\picture_analysis")
    output_dir.mkdir(exist_ok=True)

    # iTransformer Attention (Variable Correlation)
    # Using 'spring' as an example as requested before, or 'all' if available. 
    # Let's try to find any available spring/all iTransformer internals.
    it_path = base_dir / "iTransformer" / "iTransformer_spring_unified_internals.npz"
    if not it_path.exists():
        # Fallback to 'all' if spring doesn't exist (though it should based on previous steps)
        it_path = base_dir / "iTransformer" / "iTransformer_all_unified_internals.npz"

    if it_path.exists():
        data = np.load(it_path)
        attn = data['attention_map'] # (B, N, N)
        if attn.ndim == 3:
            attn = attn.mean(axis=0)
        
        labels = data['feature_names']
        plot_heatmap(attn, labels, "iTransformer: Variable-to-Variable Correlation (Attention Map)", output_dir / "iTransformer_only_heatmap.png")
    else:
        print(f"❌ Internal data not found at {it_path}")

if __name__ == "__main__":
    main()
