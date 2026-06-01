import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def make_run_id(model: str, dataset: str, seq_mode: str, strict_split: int, tag: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag2 = f"_{tag}" if tag else ""
    return f"{ts}_{model}_{dataset}_{seq_mode}_strict{int(strict_split)}{tag2}"

def prepare_run_dirs(base_dir: Path, run_id: str) -> dict:
    run_dir = base_dir / "runs" / run_id
    d = {
        "run_dir": run_dir,
        "model_dir": run_dir / "model",
        "pred_dir": run_dir / "pred",
        "plot_dir": run_dir / "plot",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d

def save_config(run_dir: Path, config: dict) -> Path:
    p = run_dir / "config.json"
    p.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

def append_summary(master_csv: Path, row: dict):
    df_row = pd.DataFrame([row])
    if master_csv.exists():
        old = pd.read_csv(master_csv)
        new = pd.concat([old, df_row], ignore_index=True)
    else:
        new = df_row
    new.to_csv(master_csv, index=False, encoding="utf-8-sig")
