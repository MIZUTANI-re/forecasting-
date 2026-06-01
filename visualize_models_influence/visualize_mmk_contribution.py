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
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Add project root


from optimiser.models import MMKModel

# --- Japanese Font Setup ---
plt.rcParams['font.family'] = 'MS Gothic'

def fmt_float(val):
    """Formats float to be meaningful (scientific if small, standard otherwise)."""
    if abs(val) == 0:
        return "0"
    if abs(val) < 0.01:
        return f"{val:.2e}"
    return f"{val:.2f}"

def fit_sinusoid(x, y):
    """Fits y = A * sin(B * x + C) + D"""
    def sin_func(x, a, b, c, d):
        return a * np.sin(b * x + c) + d
    
    p0 = [np.std(y), 1.0, 0.0, np.mean(y)]
    try:
        popt, _ = curve_fit(sin_func, x, y, p0=p0, maxfev=10000)
        y_pred = sin_func(x, *popt)
        r2 = r2_score(y, y_pred)
        
        a, b, c, d = popt
        sign_c = "+" if c >= 0 else "-"
        sign_d = "+" if d >= 0 else "-"
        
        a_str = fmt_float(a)
        b_str = fmt_float(b)
        c_str = fmt_float(abs(c))
        d_str = fmt_float(abs(d))
        
        eq_str = f"{a_str}sin({b_str}x {sign_c} {c_str}) {sign_d} {d_str}"
        return r2, eq_str, "Wavelet型"
    except:
        return -1.0, "Fit Failed", "Error"

def fit_polynomial(x, y, deg=3):
    """Fits polynomial of degree deg"""
    try:
        coeffs = np.polyfit(x, y, deg)
        p = np.poly1d(coeffs)
        y_pred = p(x)
        r2 = r2_score(y, y_pred)
        
        if deg == 3:
            a, b, c, d = coeffs
            sign_b = "+" if b >= 0 else "-"
            sign_c = "+" if c >= 0 else "-"
            sign_d = "+" if d >= 0 else "-"
            
            eq_str = f"{fmt_float(a)}x³ {sign_b} {fmt_float(abs(b))}x² {sign_c} {fmt_float(abs(c))}x {sign_d} {fmt_float(abs(d))}"
            return r2, eq_str, "Taylor/Jacobi型 (3次)"
            
        elif deg == 2:
            a, b, c = coeffs
            sign_b = "+" if b >= 0 else "-"
            sign_c = "+" if c >= 0 else "-"
            
            eq_str = f"{fmt_float(a)}x² {sign_b} {fmt_float(abs(b))}x {sign_c} {fmt_float(abs(c))}"
            return r2, eq_str, "Taylor/Jacobi型 (2次)"

        return -1.0, "Fit Failed", "Error"
    except:
        return -1.0, "Fit Failed", "Error"

def classify_and_get_eq(x, y):
    if len(set(y)) == 1: # Constant
        return f"y = {fmt_float(y[0])}", "定数"

    # Try fits
    r2_poly3, eq_poly3, name_poly3 = fit_polynomial(x, y, deg=3)
    r2_poly2, eq_poly2, name_poly2 = fit_polynomial(x, y, deg=2)
    r2_sin, eq_sin, name_sin = fit_sinusoid(x, y)
    
    threshold = 0.85
    
    # Selection Logic:
    # 1. Prefer Wavelet if it's the clear winner (or Cyclic feature)
    # 2. Prefer Poly 2 if R2 is close to Poly 3
    
    candidates = [
        (r2_sin, eq_sin, name_sin),
        (r2_poly3, eq_poly3, name_poly3),
        (r2_poly2, eq_poly2, name_poly2)
    ]
    
    # Sort by R2 descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_r2, best_eq, best_type = candidates[0]
    
    # If best fit is poor -> Spline
    if best_r2 < threshold:
        return eq_poly3 + " (近似)", "Spline型 (複合)"

    # If Poly 3 is best, check if Poly 2 is "close enough" (Occam's Razor)
    # "Close enough" = within 0.05 R2 difference?
    if best_type == "Taylor/Jacobi型 (3次)":
        diff = r2_poly3 - r2_poly2
        if diff < 0.05 and r2_poly2 > threshold:
            # Downgrade to Quadratic for simplicity
            return eq_poly2, name_poly2
            
    return best_eq, best_type

def visualize_mmk_contributions():
    # --- Configuration ---
    dim_in = 10 
    n_experts_list = [4] * dim_in
    
    config = {
        "seq_len": 72,
        "pred_len": 24, 
        "in_dim": dim_in - 1,
        "d_model": 128,
        "n_experts": n_experts_list,
        "grid_size": 3,
        "num_layers": 1
    }
    
    season_map = {
        'spring': '春 (Spring)',
        'summer': '夏 (Summer)',
        'autumn': '秋 (Autumn)',
        'winter': '冬 (Winter)'
    }
    
    base_save_dir = Path(r"c:\Users\2213144\practice\picture_analysis")
    base_save_dir.mkdir(exist_ok=True)
    equation_report_path = base_save_dir / "mmk_equations.md"

    # Cleanup existing Sin/Cos plots as per user request
    for p in base_save_dir.glob("*_Sin.png"):
        try: p.unlink()
        except: pass
    for p in base_save_dir.glob("*_Cos.png"):
        try: p.unlink()
        except: pass
    
    device = torch.device("cpu")
    
    feature_map = {
        0: "過去の電力 (正規化)",
        1: "気温",
        2: "降水量",
        3: "日射量",
        4: "曜日 (Sin)",
        5: "曜日 (Cos)",
        6: "月 (Sin)",
        7: "月 (Cos)",
        8: "祝日フラグ",
        9: "休日フラグ" # Restday
    }
    
    feature_map_en = {
        0: "Past Power Normalized",
        1: "Temperature",
        2: "Precipitation",
        3: "Solar Radiation",
        4: "Weekday Sin",
        5: "Weekday Cos",
        6: "Month Sin",
        7: "Month Cos",
        8: "Holiday Flag",
        9: "Restday Flag"
    }

    cyclic_features = [
        ("曜日 (統合)", "Weekday Unified", 4, 5, 7, ["月", "火", "水", "木", "金", "土", "日"]),
        ("月 (統合)", "Month Unified", 6, 7, 12, ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"])
    ]

    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'MS Gothic'
    sweep = np.linspace(-1.0, 1.0, 200)

    equation_report = ["# MMK Model - エキスパート機能分類レポート (日本語・簡略化版)\n"]
    equation_report.append("| 季節 | 特徴量 | コンポーネント | 推定タイプ | 近似式 |")
    equation_report.append("|---|---|---|---|---|")

    seasons_keys = ['spring', 'summer', 'autumn', 'winter']

    for season_key in seasons_keys:
        season_jp = season_map[season_key]
        print(f"\n=== Processing Season: {season_jp} ===")
        
        model_path = Path(f"c:/Users/2213144/practice/learning_saver/MMK_{season_key}_separated.pth")
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            continue

        model = MMKModel(
            seq_len=config["seq_len"],
            exog_dim=config["in_dim"],
            horizon=config["pred_len"],
            d_model=config["d_model"],
            n_experts=config["n_experts"],
            grid_size=config["grid_size"],
            num_layers=config["num_layers"]
        ).to(device)

        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        except Exception as e:
            print(f"❌ Failed to load model for {season_key}: {e}")
            continue

        # Data for Summary Plot
        feature_importance = {}

        # 1. Standard Features
        for feat_idx, feat_name_jp in feature_map.items():
            if feat_idx >= len(model.branches):
                continue
            
            feat_name_en = feature_map_en[feat_idx]
            
            # Skip individual Sin/Cos features as per user request (only show Unified)
            if "Sin" in feat_name_en or "Cos" in feat_name_en:
                continue

            branch = model.branches[feat_idx]
            experts = branch.experts if hasattr(branch, 'experts') else branch[0].experts
            
            # Binary Feature Handling
            is_binary = "Flag" in feat_name_en or "フラグ" in feat_name_jp
            if is_binary:
                # Use discrete states for flags
                # Note: This assumes inputs are somewhat close to 0/1 or the user wants to see response at 0/1.
                # If inputs are normalized, 0 and 1 might be 0-sigma and 1-sigma.
                current_sweep = np.array([0.0, 1.0])
            else:
                current_sweep = sweep
            
            total_contributions = []
            expert_outputs = [[] for _ in range(4)] 
            
            for v in current_sweep:
                inp = torch.full((1, config["seq_len"]), v, device=device).float()
                with torch.no_grad():
                    total_out = branch(inp) 
                    total_contributions.append(total_out.mean().item())
                    
                    for i, expert in enumerate(experts):
                        e_out = expert(inp)
                        expert_outputs[i].append(e_out.mean().item())
            
            # Record Importance (Range of Total Contribution)
            feature_importance[feat_name_jp] = max(total_contributions) - min(total_contributions)

            # Classify
            if is_binary:
                total_eq = f"Slope: {total_contributions[1] - total_contributions[0]:.2f}"
                total_type = "Binary Linear"
                expert_results = [] # Skip complexity for binary
            else:
                total_eq, total_type = classify_and_get_eq(current_sweep, total_contributions)
                expert_results = []
                for vals in expert_outputs:
                    eq, etype = classify_and_get_eq(current_sweep, vals)
                    expert_results.append((eq, etype))
            
            # Report
            equation_report.append(f"| {season_jp.split(' ')[0]} | {feat_name_jp} | **合計** | {total_type} | `${total_eq}$` |")
            if not is_binary:
                for i, (eq, etype) in enumerate(expert_results):
                   equation_report.append(f"| | | エキスパート {i+1} | {etype} | `${eq}$` |")

            # Plot - Single Plot for Total Only
            fig, ax0 = plt.subplots(1, 1, figsize=(12, 8))
            
            if is_binary:
                # Bar plot for 0/1
                ax0.bar(current_sweep, total_contributions, width=0.1, color='black', label=f'合計 ({total_type})')
                ax0.set_xticks([0.0, 1.0])
                ax0.set_xticklabels(['0 (Off)', '1 (On)'])
                ax0.set_title(f"[{season_jp}] {feat_name_jp}\n(Binary Feature)", fontsize=16, fontweight='bold')
            else:
                # Line plot for continuous
                ax0.plot(current_sweep, total_contributions, color='black', linewidth=3, label=f'合計 ({total_type})')
                ax0.set_title(f"[{season_jp}] {feat_name_jp}\n合計式: {total_eq}", fontsize=16, fontweight='bold')
                
            ax0.set_ylabel("予測への貢献度", fontsize=12)
            ax0.set_xlabel("入力特徴量 (値)", fontsize=12)
            ax0.legend(loc='upper right', fontsize=12)
            ax0.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            safe_name = feat_name_en.replace(" ", "_").replace("(", "").replace(")", "")
            out_path = base_save_dir / f"MMK_{season_key}_Contribution_{safe_name}.png"
            plt.savefig(out_path, dpi=100)
            plt.close()

        # 2. Unified Cyclic Plots
        print(f"   Generating Unified Cyclic Plots...")
        
        # Season to Month Indices mapping (0=Jan, 1=Feb, ..., 11=Dec)
        # Spring: 3,4,5 (Indices 2,3,4)
        # Summer: 6,7,8 (Indices 5,6,7)
        # Autumn: 9,10,11 (Indices 8,9,10)
        # Winter: 12,1,2 (Indices 11,0,1)
        season_month_map = {
            'spring': [2, 3, 4],
            'summer': [5, 6, 7],
            'autumn': [8, 9, 10],
            'winter': [11, 0, 1]
        }
        
        for name_jp, name_en, sin_idx, cos_idx, period, labels in cyclic_features:
            if "Month" in name_en:
                # Filter indices for Season
                valid_indices = season_month_map[season_key]
                t_steps = np.array(valid_indices)
                # Sort for better plotting if needed, but for Winter (11,0,1) it might be disjoint in line plot.
                # Let's just plot them as points or bars.
            else:
                t_steps = np.arange(period) # Weekday (0-6)
            
            sin_vals = np.sin(2 * np.pi * t_steps / period)
            cos_vals = np.cos(2 * np.pi * t_steps / period)
            
            branch_sin = model.branches[sin_idx]
            branch_cos = model.branches[cos_idx]
            
            total_unified = []
            with torch.no_grad():
                for s, c in zip(sin_vals, cos_vals):
                    inp_s = torch.full((1, config["seq_len"]), s, device=device).float()
                    out_s = branch_sin(inp_s).mean().item()
                    inp_c = torch.full((1, config["seq_len"]), c, device=device).float()
                    out_c = branch_cos(inp_c).mean().item()
                    total_unified.append(out_s + out_c)
            
            # Update Importance
            feature_importance[name_jp] = max(total_unified) - min(total_unified)

            plt.figure(figsize=(10, 6))
            
            # Use bar plot for discrete months/days to avoid confusion with "continuous" time
            # Or scatter with line if connected.
            # User asked "spring の際は 3,4,5 だけになります" -> implied subset.
            
            subset_labels = [labels[i] for i in t_steps]
            
            # Re-order for plot if necessary? 
            # For Winter [11, 0, 1] -> [Dec, Jan, Feb]. The x-axis might be 0,1,11.
            # Let's simple bar/scatter plot.
            plt.bar(range(len(t_steps)), total_unified, color='purple', alpha=0.7)
            plt.xticks(range(len(t_steps)), subset_labels, fontsize=12)
            
            plt.title(f"[{season_jp}] {name_jp} の合計貢献度 (Season-Specific)", fontsize=16, fontweight='bold')
            plt.ylabel("予測への貢献度 (Unit)", fontsize=12)
            plt.grid(True, axis='y', alpha=0.3)
            
            safe_name_unified = name_en.replace(" ", "_")
            out_path_unified = base_save_dir / f"MMK_{season_key}_Contribution_{safe_name_unified}.png"
            plt.savefig(out_path_unified, dpi=100)
            plt.close()

        # 3. Summary Plot (Feature Importance Bar Chart)
        print(f"   Generating Summary Importance Plot...")
        plt.figure(figsize=(12, 8))
        
        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_importance]
        values = [x[1] for x in sorted_importance]
        
        sns.barplot(x=values, y=names, palette="viridis")
        plt.title(f"[{season_jp}] 特徴量重要度 (貢献度の変動幅)", fontsize=16, fontweight='bold')
        plt.xlabel("貢献度の最大変動幅 (Range)", fontsize=12)
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        out_path_summary = base_save_dir / f"MMK_{season_key}_Feature_Importance.png"
        plt.savefig(out_path_summary, dpi=100)
        plt.close()

    # Save Report
    with open(equation_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(equation_report))
    print(f"\nEquation report saved to: {equation_report_path}")
    print("\nAll seasonal visualizations completed.")

if __name__ == "__main__":
    visualize_mmk_contributions()
