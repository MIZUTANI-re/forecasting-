
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional

# ============================================================
# Feature Definitions
# ============================================================

class FeatureConfig:
    def __init__(self, target_col: str = "power_demand"):
        self.target = target_col
        self.basic_features = [
            self.target, "temperature", "precipitation", 
            "time", "day", "solar_mean", "restday", "holiday_onehot"
        ]
        # These will be dynamically added based on dataframe columns
        self.dynamic_features = ["weekday_", "month_"]

    def get_all_features(self, df_columns: List[str]) -> List[str]:
        """Returns the full list of features present in the dataframe columns."""
        feats = []
        # Check basic features
        for b in self.basic_features:
            if b in df_columns:
                feats.append(b)
        
        # Check dynamic (one-hot or raw categorical)
        for c in df_columns:
            # One-hot versions
            is_dynamic = any(c.startswith(dyn) for dyn in self.dynamic_features)
            # Integer versions (lightweight mode)
            is_int_cat = c in ["weekday", "month"]
            # Cyclic versions
            is_cyclic = c.endswith("_sin") or c.endswith("_cos")
            
            if (is_dynamic or is_int_cat or is_cyclic) and c not in feats:
                feats.append(c)
        return feats

    def get_exog_features(self, all_features: List[str]) -> List[str]:
        """Returns all features except the target."""
        return [f for f in all_features if f != self.target]


# ============================================================
# Feature Engineering
# ============================================================

class FeatureEngineer:
    def __init__(self, sort_by_timestamp: bool = False):
        self.sort_by_timestamp = sort_by_timestamp

    def infer_step_minutes(self, df: pd.DataFrame) -> Tuple[int, int]:
        per_day = int(df.groupby(df["datetime"].dt.date)["time"].nunique().mode().iloc[0])
        if 1440 % per_day == 0:
            step_minutes = 1440 // per_day
        else:
            tmax = int(df["time"].max())
            step_minutes = 60 if tmax <= 23 else 30
        return per_day, step_minutes

    def build_timestamp(self, df: pd.DataFrame) -> Tuple[pd.Series, bool, int, int]:
        dt = pd.to_datetime(df["datetime"])
        has_time_component = ((dt.dt.hour != 0) | (dt.dt.minute != 0) | (dt.dt.second != 0)).any()

        per_day, step_minutes = self.infer_step_minutes(df)
        if has_time_component:
            ts = dt
        else:
            ts = dt + pd.to_timedelta(df["time"] * step_minutes, unit="m")
        return ts, has_time_component, per_day, step_minutes

    def preprocess(self, df: pd.DataFrame, use_onehot: bool = True, use_cyclic: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Applies standard preprocessing:
        1. Datetime conversion
        2. Sorting
        3. Timestamp creation
        4. Calendar features (year, month, day, weekday)
        5. Encoding: One-hot (default), Cyclic (sin/cos), or Integer
        """
        df = df.copy()
        
        # Basic validation
        if "time" not in df.columns:
            raise ValueError("CSV must contain 'time' column.")
            
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        # Sort by datetime + time to ensure correct order
        df = df.sort_values(["datetime", "time"]).reset_index(drop=True)

        # Build timestamp
        ts, has_time, per_day, step_min = self.build_timestamp(df)
        df["timestamp"] = ts
        
        if self.sort_by_timestamp:
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Calendar features
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day"] = df["timestamp"].dt.day
        df["weekday"] = df["timestamp"].dt.weekday

        # Drop NaNs
        df = df.dropna().reset_index(drop=True)

        # Encoding Logic
        if use_cyclic:
            # 1. Weekday (0-6) -> 7 days cycle
            df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
            df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
            
            # 2. Month (1-12) -> 12 months cycle
            df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
            df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
            
            # Remove raw features to avoid duplicity/leakage in models that don't need them
            df.drop(columns=["weekday", "month"], inplace=True)
            print("🔄 Using Cyclic Encoding for categorical features (Sin/Cos)")

        elif use_onehot:
            df = pd.get_dummies(df, columns=["weekday", "month"], drop_first=True)
        else:
            # Shift month to 0-indexed for consistency if needed, but 1-12 is fine for KAN
            print("🚀 Using Integer Encoding for categorical features (Lightweight Mode)")

        metadata = {
            "per_day": per_day,
            "step_minutes": step_min,
            "has_time_component": has_time,
            "use_onehot": use_onehot,
            "use_cyclic": use_cyclic
        }
        
        return df, metadata


# ============================================================
# Scaling
# ============================================================

class ScalerWrapper:
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, features: List[str], target: str):
        self.feature_scaler.fit(df[features])
        self.target_scaler.fit(df[[target]])

    def transform(self, df: pd.DataFrame, features: List[str], target: str) -> Tuple[np.ndarray, np.ndarray]:
        X = self.feature_scaler.transform(df[features])
        y = self.target_scaler.transform(df[[target]])
        return X, y

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform 1D or 2D target array.
        If 1D, reshapes locally for transform then flattens back.
        """
        shape = y_scaled.shape
        if len(shape) == 1:
            y_2d = y_scaled.reshape(-1, 1)
            y_inv = self.target_scaler.inverse_transform(y_2d)
            return y_inv.flatten()
        elif len(shape) == 2:
            return self.target_scaler.inverse_transform(y_scaled)
        else:
            # For 3D+ (e.g. Batch x Horizon), inverse by last dim
            # But StandardScaler expects 2D (Sample, Feature).
            # Here target dim is 1, so we handle accordingly.
            flat = y_scaled.reshape(-1, 1)
            inv = self.target_scaler.inverse_transform(flat)
            return inv.reshape(shape)

    def inverse_1col(self, y2d: np.ndarray) -> np.ndarray:
        """Helper for simpler inverse (manual calculation) if needed."""
        return y2d * self.target_scaler.scale_[0] + self.target_scaler.mean_[0]
