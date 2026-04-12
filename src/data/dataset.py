from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


EPS = 1e-8


class NIFTY100Dataset(Dataset):
    """
    Returns one sample = one (date, all_stocks) snapshot.

    Item dict:
      features:    Tensor [N, L, F]
      targets_reg: Tensor [N]
      targets_dir: Tensor [N]
      target_mask: Tensor [N]     1 where target exists for date/stock else 0
      regime:      Tensor [R]
      sector_ids:  LongTensor [N]
      date:        str
    """

    def __init__(self, config: DictConfig, date_spec=None):
        self.config = config
        self.lookback = int(config.features.lookback)
        self.primary_horizon = int(config.features.primary_target)
        self.corr_window = int(config.model.corr_window)

        processed_dir = Path(config.data.processed_dir)
        universe = pd.read_csv(config.data.universe_file)
        all_tickers = universe["ticker"].astype(str).tolist()

        available_tickers = []
        ticker_frames = {}
        for ticker in all_tickers:
            fp = processed_dir / f"{ticker}_features.parquet"
            if fp.exists():
                df = pd.read_parquet(fp)
                df["Date"] = pd.to_datetime(df["Date"])
                ticker_frames[ticker] = df.sort_values("Date")
                available_tickers.append(ticker)

        if not available_tickers:
            raise RuntimeError("No processed ticker feature files found in data/processed")

        self.tickers = available_tickers
        self.n_stocks = len(self.tickers)

        feature_json = processed_dir / "feature_columns.json"
        if feature_json.exists():
            with feature_json.open("r", encoding="utf-8") as fp:
                feature_cols = json.load(fp)
        else:
            sample_cols = ticker_frames[self.tickers[0]].columns.tolist()
            feature_cols = [
                c
                for c in sample_cols
                if c not in {"Date", "ticker"} and not c.startswith("target_")
            ]
        self.feature_columns = feature_cols

        regime_fp = processed_dir / "regime_features.parquet"
        if not regime_fp.exists():
            raise FileNotFoundError("Missing data/processed/regime_features.parquet")
        regime_df = pd.read_parquet(regime_fp)
        regime_df["Date"] = pd.to_datetime(regime_df["Date"])
        regime_df = regime_df.sort_values("Date")
        self.regime_cols = [c for c in regime_df.columns if c != "Date"]

        sector_fp = processed_dir / "sector_map.parquet"
        if sector_fp.exists():
            sector_df = pd.read_parquet(sector_fp)
        else:
            sector_df = universe[["ticker", "sector"]].copy()

        sector_df = sector_df[sector_df["ticker"].isin(self.tickers)].copy()
        unique_sectors = sorted(sector_df["sector"].dropna().astype(str).unique().tolist())
        self.sector_to_id = {s: i for i, s in enumerate(unique_sectors)}
        sector_lookup = dict(zip(sector_df["ticker"], sector_df["sector"]))
        self.sector_ids = np.array([self.sector_to_id.get(sector_lookup[t], 0) for t in self.tickers], dtype=np.int64)

        # Use regime calendar as the master market calendar.
        # Per-stock missing dates are imputed in features and masked in targets.
        self.calendar = pd.DatetimeIndex(pd.to_datetime(regime_df["Date"]).sort_values().unique())
        if len(self.calendar) == 0:
            raise RuntimeError("No dates found in regime_features.parquet")
        t_len = len(self.calendar)

        self.features_arr = np.zeros((self.n_stocks, t_len, len(self.feature_columns)), dtype=np.float32)
        self.targets_reg = np.full((self.n_stocks, t_len), np.nan, dtype=np.float32)
        self.targets_dir = np.full((self.n_stocks, t_len), np.nan, dtype=np.float32)
        self.target_mask = np.zeros((self.n_stocks, t_len), dtype=np.float32)
        self.log_rets = np.zeros((self.n_stocks, t_len), dtype=np.float32)

        for i, ticker in enumerate(self.tickers):
            aligned = ticker_frames[ticker].set_index("Date").reindex(self.calendar)
            feat = aligned[self.feature_columns].to_numpy(dtype=np.float32)
            self.features_arr[i] = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

            target_reg_col = f"target_res_{self.primary_horizon}"
            target_dir_col = f"target_dir_{self.primary_horizon}"
            self.targets_reg[i] = aligned[target_reg_col].to_numpy(dtype=np.float32)
            self.targets_dir[i] = aligned[target_dir_col].to_numpy(dtype=np.float32)

            m = np.isfinite(self.targets_reg[i]) & np.isfinite(self.targets_dir[i])
            self.target_mask[i] = m.astype(np.float32)

            if "log_ret" in aligned.columns:
                lr = aligned["log_ret"].to_numpy(dtype=np.float32)
                self.log_rets[i] = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)

        regime_aligned = regime_df.set_index("Date").reindex(self.calendar)
        self.regime_arr = regime_aligned[self.regime_cols].to_numpy(dtype=np.float32)
        self.regime_arr = np.nan_to_num(self.regime_arr, nan=0.0, posinf=0.0, neginf=0.0)

        valid_idxs = []
        min_required_targets = max(2 * int(config.evaluation.top_k), 1)
        for idx in range(t_len):
            if idx < self.lookback - 1:
                continue
            if idx < self.corr_window - 1:
                continue
            available = int(self.target_mask[:, idx].sum())
            if available < min_required_targets:
                continue
            valid_idxs.append(idx)

        self.valid_idxs = self._filter_by_date_spec(valid_idxs, date_spec)

    def _filter_by_date_spec(self, idxs: list[int], date_spec) -> list[int]:
        if date_spec is None:
            return idxs

        if isinstance(date_spec, tuple) and len(date_spec) == 2:
            start, end = pd.Timestamp(date_spec[0]), pd.Timestamp(date_spec[1])
            return [i for i in idxs if start <= self.calendar[i] <= end]

        if isinstance(date_spec, (list, tuple, set, pd.Index, np.ndarray)):
            date_set = {pd.Timestamp(d) for d in date_spec}
            return [i for i in idxs if self.calendar[i] in date_set]

        return idxs

    def __len__(self) -> int:
        return len(self.valid_idxs)

    def __getitem__(self, index: int):
        t_idx = self.valid_idxs[index]

        x = self.features_arr[:, t_idx - self.lookback + 1 : t_idx + 1, :]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        rr = self.log_rets[:, t_idx - self.corr_window + 1 : t_idx + 1]
        rr = np.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0)

        target_reg_raw = self.targets_reg[:, t_idx]
        target_dir_raw = self.targets_dir[:, t_idx]
        target_mask = self.target_mask[:, t_idx]

        target_reg = np.nan_to_num(target_reg_raw, nan=0.0, posinf=0.0, neginf=0.0)
        target_dir = np.nan_to_num(target_dir_raw, nan=0.0, posinf=0.0, neginf=0.0)

        sample = {
            "features": torch.tensor(x, dtype=torch.float16),
            "targets_reg": torch.tensor(target_reg, dtype=torch.float32),
            "targets_dir": torch.tensor(target_dir, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.float32),
            "regime": torch.tensor(self.regime_arr[t_idx], dtype=torch.float16),
            "sector_ids": torch.tensor(self.sector_ids, dtype=torch.long),
            "raw_returns": torch.tensor(rr, dtype=torch.float32),
            "date": self.calendar[t_idx].strftime("%Y-%m-%d"),
        }
        return sample


def build_dataloaders(config: DictConfig, split):
    """
    Build DataLoader for train/val/test split or explicit date spec.

    split can be one of:
      - "train" / "val" / "test" where config.runtime_split_dates[split] is available
      - list of dates
      - (start_date, end_date)
    """
    date_spec = split
    shuffle = False

    if isinstance(split, str):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split string: {split}")
        runtime_splits = getattr(config, "runtime_split_dates", None)
        if runtime_splits is None or split not in runtime_splits:
            raise ValueError("For split='train'/'val'/'test', config.runtime_split_dates must be populated")
        date_spec = list(runtime_splits[split])
        shuffle = split == "train"
    else:
        shuffle = False

    ds = NIFTY100Dataset(config=config, date_spec=date_spec)
    loader = DataLoader(
        ds,
        batch_size=int(config.training.batch_dates),
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    return loader
