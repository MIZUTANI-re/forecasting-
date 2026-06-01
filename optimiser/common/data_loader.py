# practice/optimiser/common/data_loader.py
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =========================
# Feature policy
# =========================
@dataclass
class FeaturePolicy:
    """
    - mode="all_in": Xに power_demand も含めて全部入れる（Transformer/iTransformer向け）
    - mode="split":  X_exog と y を分ける（Hybrid/Integrated系向け）
    """
    mode: str = "split"  # "all_in" or "split"

    base_exog: Tuple[str, ...] = (
        "temperature",
        "precipitation",
        "time",
        "day",
        "solar_mean",
        "restday",
        "holiday_onehot",
    )

    target_col: str = "power_demand"


# =========================
# Timestamp helpers
# =========================
def infer_step_minutes(df: pd.DataFrame) -> Tuple[int, int]:
    per_day = int(df.groupby(df["datetime"].dt.date)["time"].nunique().mode().iloc[0])
    if 1440 % per_day == 0:
        step_minutes = 1440 // per_day
    else:
        tmax = int(df["time"].max())
        step_minutes = 60 if tmax <= 23 else 30
    return int(per_day), int(step_minutes)


def build_timestamp(df: pd.DataFrame) -> Tuple[pd.Series, bool, int, int]:
    dt = pd.to_datetime(df["datetime"])
    has_time_component = ((dt.dt.hour != 0) | (dt.dt.minute != 0) | (dt.dt.second != 0)).any()
    per_day, step_minutes = infer_step_minutes(df)
    if has_time_component:
        ts = dt
    else:
        ts = dt + pd.to_timedelta(df["time"].astype(float) * step_minutes, unit="m")
    return ts, bool(has_time_component), int(per_day), int(step_minutes)


# =========================
# Sequence builders
# =========================
def make_sequences_allin(X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int):
    Xs, Ys, starts = [], [], []
    T = len(y)
    for i in range(T - seq_len - horizon + 1):
        Xs.append(X[i:i + seq_len])
        Ys.append(y[i + seq_len:i + seq_len + horizon].reshape(-1))
        starts.append(i)
    return np.array(Xs), np.array(Ys), np.array(starts, dtype=np.int64)


def make_sequences_split(exog: np.ndarray, y: np.ndarray, seq_len: int, horizon: int):
    X_ex, Y_past, Y_fut, starts = [], [], [], []
    T = len(y)
    for i in range(T - seq_len - horizon + 1):
        X_ex.append(exog[i:i + seq_len])
        Y_past.append(y[i:i + seq_len])
        Y_fut.append(y[i + seq_len:i + seq_len + horizon].reshape(-1))
        starts.append(i)
    return np.array(X_ex), np.array(Y_past), np.array(Y_fut), np.array(starts, dtype=np.int64)


def strict_split_by_y_window(starts: np.ndarray, n_raw: int, tr: int, va: int, seq_len: int, horizon: int):
    y_start = starts + seq_len
    y_end = y_start + horizon - 1
    m_tr = (y_end < tr)
    m_va = (y_start >= tr) & (y_end < va)
    m_te = (y_start >= va) & (y_end < n_raw)
    return m_tr, m_va, m_te


# =========================
# Main prepare function
# =========================
def prepare_data(
    file_path: str,
    seq_len: int,
    horizon: int,
    policy: FeaturePolicy,
    sequence_mode: str = "warm",
    strict_split: bool = True,
    sort_by_timestamp: bool = False,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Dict[str, object]:

    df = pd.read_csv(file_path)

    for c in ["datetime", "time", policy.target_col]:
        if c not in df.columns:
            raise ValueError(f"必須列 '{c}' がありません。列={list(df.columns)}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["time"] = df["time"].astype(int)

    df = df.sort_values(["datetime", "time"]).reset_index(drop=True)
    df["timestamp"], has_time_component, per_day, step_minutes = build_timestamp(df)

    if sort_by_timestamp:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # calendar
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday

    df = df.dropna().reset_index(drop=True)
    df = pd.get_dummies(df, columns=["weekday", "month"], drop_first=True)

    oh_cols = [c for c in df.columns if c.startswith("weekday_") or c.startswith("month_")]
    exog_cols = list(policy.base_exog) + oh_cols

    n = len(df)
    tr = int(n * split_ratio[0])
    va = int(n * (split_ratio[0] + split_ratio[1]))

    all_ts = df["timestamp"].reset_index(drop=True)
    all_time = df["time"].reset_index(drop=True)

    # scalerは train point のみfit（リーク防止）
    sy = StandardScaler().fit(df.iloc[:tr][[policy.target_col]])

    if policy.mode == "all_in":
        X_cols = [policy.target_col] + exog_cols
        sX = StandardScaler().fit(df.iloc[:tr][X_cols])
        X_all = sX.transform(df[X_cols])
        y_all = sy.transform(df[[policy.target_col]])

        if sequence_mode != "warm":
            raise NotImplementedError("all_in + cold は必要なら後で追加でOK")

        X_seq, Y_seq, starts = make_sequences_allin(X_all, y_all, seq_len, horizon)

        if strict_split:
            m_tr, m_va, m_te = strict_split_by_y_window(starts, len(y_all), tr, va, seq_len, horizon)
            Xtr, Ytr = X_seq[m_tr], Y_seq[m_tr]
            Xva, Yva = X_seq[m_va], Y_seq[m_va]
            Xte, Yte = X_seq[m_te], Y_seq[m_te]
            st_te = starts[m_te]
        else:
            train_end = tr - seq_len - horizon + 1
            val_end = va - seq_len - horizon + 1
            Xtr, Ytr = X_seq[:train_end], Y_seq[:train_end]
            Xva, Yva = X_seq[train_end:val_end], Y_seq[train_end:val_end]
            Xte, Yte = X_seq[val_end:], Y_seq[val_end:]
            st_te = starts[val_end:]

        return {
            "mode": "all_in",
            "Xtr": Xtr, "Ytr": Ytr,
            "Xva": Xva, "Yva": Yva,
            "Xte": Xte, "Yte": Yte,
            "st_te": st_te,
            "scaler_y": sy,
            "per_day": per_day, "step_minutes": step_minutes, "has_time_component": has_time_component,
            "all_ts": all_ts, "all_time": all_time,
            "file_basename": os.path.basename(file_path),
            "feature_cols": X_cols,
        }

    elif policy.mode == "split":
        sx = StandardScaler().fit(df.iloc[:tr][exog_cols])
        X_exog_all = sx.transform(df[exog_cols])
        y_all = sy.transform(df[[policy.target_col]])

        if sequence_mode != "warm":
            raise NotImplementedError("split + cold は必要なら後で追加でOK")

        X_ex, Yp, Yf, starts = make_sequences_split(X_exog_all, y_all, seq_len, horizon)

        if strict_split:
            m_tr, m_va, m_te = strict_split_by_y_window(starts, len(y_all), tr, va, seq_len, horizon)
            Xtr, Yptr, Yftr = X_ex[m_tr], Yp[m_tr], Yf[m_tr]
            Xva, Ypva, Yfva = X_ex[m_va], Yp[m_va], Yf[m_va]
            Xte, Ypte, Yfte = X_ex[m_te], Yp[m_te], Yf[m_te]
            st_te = starts[m_te]
        else:
            train_end = tr - seq_len - horizon + 1
            val_end = va - seq_len - horizon + 1
            Xtr, Yptr, Yftr = X_ex[:train_end], Yp[:train_end], Yf[:train_end]
            Xva, Ypva, Yfva = X_ex[train_end:val_end], Yp[train_end:val_end], Yf[train_end:val_end]
            Xte, Ypte, Yfte = X_ex[val_end:], Yp[val_end:], Yf[val_end:]
            st_te = starts[val_end:]

        return {
            "mode": "split",
            "Xtr_exog": Xtr, "Yptr": Yptr, "Yftr": Yftr,
            "Xva_exog": Xva, "Ypva": Ypva, "Yfva": Yfva,
            "Xte_exog": Xte, "Ypte": Ypte, "Yfte": Yfte,
            "st_te": st_te,
            "scaler_y": sy,
            "per_day": per_day, "step_minutes": step_minutes, "has_time_component": has_time_component,
            "all_ts": all_ts, "all_time": all_time,
            "file_basename": os.path.basename(file_path),
            "exog_cols": exog_cols,
            "exog_dim": int(Xtr.shape[2]),
        }

    else:
        raise ValueError(f"policy.mode must be 'all_in' or 'split'. got={policy.mode}")


__all__ = ["FeaturePolicy", "prepare_data"]
