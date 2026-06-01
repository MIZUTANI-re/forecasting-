# このテンプレをコピーして各モデル用run_*.pyを作る
from pathlib import Path
import pandas as pd
from optimiser.common.io_utils import make_run_id, prepare_run_dirs, save_config, append_summary

BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = BASE_DIR / "result_csv"
SUMMARY_MASTER = RESULT_DIR / "summary_master.csv"

def main():
    model_name = "YourModel"
    dataset = "all"
    seq_mode = "warm"
    strict_split = 1
    run_id = make_run_id(model_name, dataset, seq_mode, strict_split, tag="test")

    dirs = prepare_run_dirs(BASE_DIR, run_id)
    config_path = save_config(dirs["run_dir"], {
        "model": model_name,
        "dataset": dataset,
        "seq_mode": seq_mode,
        "strict_split": strict_split,
    })

    # ここで学習→評価→pred_csv/fig/model保存
    # dirs["pred_dir"], dirs["plot_dir"], dirs["model_dir"] を使う

    row = {
        "date_run": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "model": model_name,
        "dataset": dataset,
        "seq_mode": seq_mode,
        "strict_split": strict_split,
        "config_path": str(config_path),
    }
    append_summary(SUMMARY_MASTER, row)
    print("DONE:", run_id)

if __name__ == "__main__":
    main()
