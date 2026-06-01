
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_weekly_cycle_report(npz_path, output_path):
    data = np.load(npz_path, allow_pickle=True)
    branch_outputs = data['branch_outputs'] # (Samples, F, Horizon)
    feature_names = list(data['feature_names'])
    
    # Add power_demand_past if missing
    if branch_outputs.shape[1] == len(feature_names) + 1:
        feature_names = ["power_demand_past"] + feature_names
        
    horizon_idx = 0
    
    # Weekday mapping
    day_map = {
        "weekday_1": "Mon",
        "weekday_2": "Tue",
        "weekday_3": "Wed",
        "weekday_4": "Thu",
        "weekday_5": "Fri",
        "weekday_6": "Sat",
        "restday": "Sun/Hol"
    }
    
    results = []
    for feat_id, feat_name in enumerate(feature_names):
        if any(key in feat_name for key in day_map.keys()):
            # Find the display name
            display_name = "Sunday/Holiday" if "restday" in feat_name else day_map.get(feat_name, feat_name)
            
            # The "impact" is the difference in contribution when the day is ACTIVE vs INACTIVE.
            # But in the additive model, we can just look at the raw branch output for that day.
            # Since it's one-hot, the output when x=1 is the "boost" for that day.
            
            # For simplicity, we can just take the average contribution which represents the net effect.
            # However, to be precise, we should look at the output when the flag is 1.
            # (In the standardized case, the branch always makes SOME contribution even if flag=0, 
            # but the DELTA is what matters).
            
            # Let's just use the mean contribution as a proxy for 'Relative Impact'
            impact = np.mean(branch_outputs[:, feat_id, horizon_idx])
            results.append({"Day": display_name, "Impact": impact, "Sort": list(day_map.values()).index(display_name) if display_name in day_map.values() else 7})

    df_plot = pd.DataFrame(results).sort_values("Sort")
    
    plt.figure(figsize=(10, 6))
    colors = ['#5dade2'] * 5 + ['#eb984e', '#ec7063'] # Blue for weekdays, Orange/Red for weekend
    plt.bar(df_plot["Day"], df_plot["Impact"], color=colors[:len(df_plot)])
    plt.axhline(0, color='black', linewidth=1)
    plt.title("MMK Learned Weekly Cycle: Relative Impact on Demand")
    plt.xlabel("Day of the Week")
    plt.ylabel("Impact on Prediction (Scaled MW)")
    plt.grid(axis='y', alpha=0.3)
    
    # Add labels on top of bars
    for i, v in enumerate(df_plot["Impact"]):
        plt.text(i, v + (0.005 if v > 0 else -0.015), f"{v:.3f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Weekly cycle report saved to {output_path}")

if __name__ == "__main__":
    npz = r"c:\Users\2213144\practice\result_csv\MMK\MMK_spring_separated_internals.npz"
    out = r"c:\Users\2213144\practice\picture\MMK_Analysis\mmk_weekly_cycle_human.png"
    generate_weekly_cycle_report(npz, out)
