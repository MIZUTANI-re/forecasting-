
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from optimiser.features import FeatureConfig, FeatureEngineer, ScalerWrapper

# ============================================================
# Datasets
# ============================================================

class UnifiedTimeSeriesDataset(Dataset):
    """
    Dataset for models taking (X, y) where X includes all features.
    Used by: iTransformer, HTMformer
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SeparatedTimeSeriesDataset(Dataset):
    """
    Dataset for models taking (X_exog, y_past, y_future).
    Used by: GridTST, IntegratedHQ, Hybrid models
    """
    def __init__(self, X_ex: np.ndarray, y_p: np.ndarray, y_f: np.ndarray):
        self.X_ex = torch.tensor(X_ex, dtype=torch.float32) # (L, exog_dim)
        self.y_p  = torch.tensor(y_p,  dtype=torch.float32) # (L, 1)
        self.y_f  = torch.tensor(y_f,  dtype=torch.float32) # (H)

    def __len__(self):
        return len(self.X_ex)

    def __getitem__(self, idx):
        return self.X_ex[idx], self.y_p[idx], self.y_f[idx]


# ============================================================
# Data Loader Factory
# ============================================================

class DataLoaderFactory:
    def __init__(self, 
                 seq_len: int = 72, 
                 horizon: int = 24, 
                 train_ratio: float = 0.8, 
                 val_ratio: float = 0.1,
                 strict_split: bool = True,
                 sequence_mode: str = "warm",
                 target_col: str = "power_demand",
                 use_onehot: bool = True,
                 use_cyclic: bool = False):
        self.seq_len = seq_len
        self.horizon = horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.strict_split = strict_split
        self.sequence_mode = sequence_mode
        self.use_onehot = use_onehot
        self.use_cyclic = use_cyclic
        self.feature_config = FeatureConfig(target_col=target_col)
        self.engineer = FeatureEngineer()
        self.scaler = ScalerWrapper()

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates basic sequences (X, y)."""
        Xs, ys = [], []
        for i in range(len(X) - self.seq_len - self.horizon + 1):
            Xs.append(X[i : i + self.seq_len])
            ys.append(y[i + self.seq_len : i + self.seq_len + self.horizon].flatten())
        return np.array(Xs), np.array(ys)

    def _create_sequences_separated(self, X_ex: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Creates separated sequences (X_ex, y_past, y_future)."""
        Xs, yps, yfs = [], [], []
        for i in range(len(y) - self.seq_len - self.horizon + 1):
            Xs.append(X_ex[i : i + self.seq_len])
            yps.append(y[i : i + self.seq_len])
            yfs.append(y[i + self.seq_len : i + self.seq_len + self.horizon].flatten())
        return np.array(Xs), np.array(yps), np.array(yfs)

    def _split_strict(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns boolean masks for train/val/test based on 'strict' splitting logic.
        Ensures y-window does not leak across splits.
        """
        starts = np.arange(n_samples)
        y_start = starts + self.seq_len
        y_end   = y_start + self.horizon - 1
        
        # Calculate raw indices for split boundaries
        # Note: These 'tr' and 'va' are indices in the *original* time series
        # We need the total length of the original data to calculate them correctly.
        # However, here n_samples is usually len(sequences).
        # Wait, strictly speaking, tr/va should be calculated on the raw dataframe length.
        # But if we are in 'warm' mode, we have access to the whole concatenated arrays.
        # Let's assume n_samples here maps roughly to the dataframe length if seq_len is small relative to N.
        # Better approach: We need the original N to calculate tr/va.
        pass # Implemented inside load_data where N is known.
        
    def load_data(self, 
                  file_path: str, 
                  mode: str = "unified", 
                  batch_size: int = 32, 
                  season: str = "all", 
                  custom_dates: Optional[Dict[str, str]] = None,
                  exclude_dates: Optional[List[Dict[str, str]]] = None):
        """
        Main entry point.
        mode: "unified" (X, y) or "separated" (X_ex, y_p, y_f)
        custom_dates: Dictionary with keys 'train_start', 'train_end', etc.
        exclude_dates: List of dictionaries with 'start' and 'end' keys to filter out from Train/Val.
        """
        # 1. Load and Preprocess
        df = pd.read_csv(file_path)
        df, meta = self.engineer.preprocess(df, use_onehot=self.use_onehot, use_cyclic=self.use_cyclic)
        
        # 1.5 Filter by Season
        if season != "all":
            # Ensure datetime is parsed
            if "datetime" in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                     df["datetime"] = pd.to_datetime(df["datetime"])
                
                season_map = {
                    "spring": [3, 4, 5],
                    "summer": [6, 7, 8],
                    "autumn": [9, 10, 11],
                    "winter": [12, 1, 2]
                }
                if season in season_map:
                    months = season_map[season]
                    original_len = len(df)
                    df = df[df["datetime"].dt.month.isin(months)].copy()
                    print(f"❄ Filtered data for season '{season}': {original_len} -> {len(df)} rows")

        
        # 2. Identify Features
        all_feats = self.feature_config.get_all_features(df.columns)
        exog_feats = self.feature_config.get_exog_features(all_feats)
        target = self.feature_config.target

        # 3. Handle Splitting Indices (Determine masks)
        n = len(df)
        if custom_dates:
            if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                df["datetime"] = pd.to_datetime(df["datetime"])
            
            ts_start = df["datetime"].min()
            ts_end = df["datetime"].max()
            
            d_tr_e = pd.to_datetime(custom_dates.get('train_end', '2020-04-30'))
            d_va_s = pd.to_datetime(custom_dates.get('val_start', d_tr_e + pd.Timedelta(hours=1)))
            d_va_e = pd.to_datetime(custom_dates.get('val_end', d_va_s + pd.Timedelta(days=30)))
            d_te_s = pd.to_datetime(custom_dates.get('test_start', '2021-01-01'))
            d_te_e = pd.to_datetime(custom_dates.get('test_end', '2021-12-31'))

            tr_idx = df[df["datetime"] <= d_tr_e].index[-1] + 1
            va_start_idx = df[df["datetime"] >= d_va_s].index[0]
            va_end_idx = df[df["datetime"] <= d_va_e].index[-1] + 1
            te_start_idx = df[df["datetime"] >= d_te_s].index[0]
            te_end_idx = df[df["datetime"] <= d_te_e].index[-1] + 1
            
            # Use specific partitions for fitting scaler to avoid leakage, but transform full data
            train_df_for_scaler = df.iloc[:tr_idx].copy()
        else:
            tr_idx = int(n * self.train_ratio)
            va_end_idx = int(n * (self.train_ratio + self.val_ratio))
            va_start_idx = tr_idx
            te_start_idx = va_end_idx
            te_end_idx = n - 1
            train_df_for_scaler = df.iloc[:tr_idx].copy()

        # 4. Standard Scaling (Fit on Train, Transform ALL)
        self.scaler.fit(train_df_for_scaler, all_feats, target)
        X_all_norm, y_all_norm = self.scaler.transform(df, all_feats, target)

        # 5. Sequence Generation & Masking
        if mode == "unified":
            # Create all possible sequences from the continuous data
            seq_X_all, seq_y_all = self._create_sequences(X_all_norm, y_all_norm)
            
            # Shift indices because sequences are aligned with their starts
            starts = np.arange(len(seq_y_all))
            y_start = starts + self.seq_len # Index where prediction starts in df
            y_end   = y_start + self.horizon - 1 # Index where prediction ends in df
            
            if custom_dates:
                m_tr = (y_end < tr_idx)
                m_va = (y_start >= va_start_idx) & (y_end < va_end_idx)
                m_te = (y_start >= te_start_idx) & (y_end <= te_end_idx)
            else:
                m_tr = (y_end < tr_idx)
                m_va = (y_start >= tr_idx) & (y_end < va_end_idx)
                m_te = (y_start >= va_end_idx)
            
            # 5.5 Optional Exclusion (Apply only to Train/Val)
            if exclude_dates:
                if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                    df["datetime"] = pd.to_datetime(df["datetime"])
                
                # Check for EACH sequence if its target window (y) overlaps with excluded periods
                # We use the midpoint of the target window for simplicity or check range
                excluded_mask = np.zeros(len(starts), dtype=bool)
                for period in exclude_dates:
                    p_start = pd.to_datetime(period['start'])
                    p_end   = pd.to_datetime(period['end'])
                    
                    # Logic: If y_start or y_end falls into the exclusion window
                    overlap = (df["datetime"].iloc[y_start].values >= p_start.to_datetime64()) & \
                              (df["datetime"].iloc[y_end].values <= p_end.to_datetime64())
                    excluded_mask = excluded_mask | overlap
                
                # Remove from all masks (Train, Val, AND Test per user request)
                m_tr = m_tr & (~excluded_mask)
                m_va = m_va & (~excluded_mask)
                m_te = m_te & (~excluded_mask)
                
            train_ds = UnifiedTimeSeriesDataset(seq_X_all[m_tr], seq_y_all[m_tr])
            val_ds   = UnifiedTimeSeriesDataset(seq_X_all[m_va], seq_y_all[m_va])
            test_ds  = UnifiedTimeSeriesDataset(seq_X_all[m_te], seq_y_all[m_te])
            test_starts = starts[m_te]

        elif mode == "separated":
            feat_to_idx = {f: i for i, f in enumerate(all_feats)}
            exog_indices = [feat_to_idx[f] for f in exog_feats]
            X_ex_all_norm = X_all_norm[:, exog_indices]
            
            seq_X_all, seq_yp_all, seq_yf_all = self._create_sequences_separated(X_ex_all_norm, y_all_norm)
            
            starts = np.arange(len(seq_yf_all))
            y_start = starts + self.seq_len
            y_end   = y_start + self.horizon - 1
            
            if custom_dates:
                m_tr = (y_end < tr_idx)
                m_va = (y_start >= va_start_idx) & (y_end < va_end_idx)
                m_te = (y_start >= te_start_idx) & (y_end <= te_end_idx)
            else:
                m_tr = (y_end < tr_idx)
                m_va = (y_start >= tr_idx) & (y_end < va_end_idx)
                m_te = (y_start >= va_end_idx)
            
            # 5.5 Optional Exclusion (Apply only to Train/Val)
            if exclude_dates:
                excluded_mask = np.zeros(len(starts), dtype=bool)
                for period in exclude_dates:
                    p_start = pd.to_datetime(period['start'])
                    p_end   = pd.to_datetime(period['end'])
                    overlap = (df["datetime"].iloc[y_start].values >= p_start.to_datetime64()) & \
                              (df["datetime"].iloc[y_end].values <= p_end.to_datetime64())
                    excluded_mask = excluded_mask | overlap
                
                m_tr = m_tr & (~excluded_mask)
                m_va = m_va & (~excluded_mask)
                m_te = m_te & (~excluded_mask)
            
            train_ds = SeparatedTimeSeriesDataset(seq_X_all[m_tr], seq_yp_all[m_tr], seq_yf_all[m_tr])
            val_ds   = SeparatedTimeSeriesDataset(seq_X_all[m_va], seq_yp_all[m_va], seq_yf_all[m_va])
            test_ds  = SeparatedTimeSeriesDataset(seq_X_all[m_te], seq_yp_all[m_te], seq_yf_all[m_te])
            test_starts = starts[m_te]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 6. Create Loaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        info = {
            "test_starts": test_starts,
            "scaler": self.scaler,
            "all_timestamp": df["timestamp"], # Full info for reconstruction
            "all_time": df["time"],
            "metadata": meta,
            "exog_dim": train_ds.X_ex.shape[2] if mode == "separated" else 0,
            "in_dim": train_ds.X.shape[2] if mode == "unified" else 0,
            "target_idx": all_feats.index(target) if target in all_feats else -1,
            "feature_names": all_feats if mode == "unified" else exog_feats,
            "tr": tr_idx,
            "va": va_end_idx
        }

        return train_loader, val_loader, test_loader, info
