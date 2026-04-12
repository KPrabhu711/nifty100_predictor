"""
Usage: python scripts/02_build_features.py

1. For each ticker in universe, load raw parquet
2. Compute feature set
3. Cross-sectionally normalize features across NIFTY 100 at each date
4. Save processed feature parquets
5. Log feature count, date range, NaN statistics
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from rich.progress import track

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import build_feature_artifacts, save_feature_artifacts
from src.data.normalization import normalize_cross_section
from src.utils.logging_utils import setup_logger


def main():
    t0 = time.time()
    config = OmegaConf.load(PROJECT_ROOT / "config" / "config.yaml")
    logger = setup_logger(config)

    artifacts = build_feature_artifacts(config, logger=logger)
    feature_frames = artifacts.feature_frames
    feature_cols = artifacts.feature_columns
    norm_feature_cols = [c for c in feature_cols if c != "vol_zscore_cross"]

    indexed = {t: df.copy().set_index("Date").sort_index() for t, df in feature_frames.items()}
    normalized = {t: df.copy() for t, df in indexed.items()}

    all_dates = sorted(set().union(*[set(df.index) for df in indexed.values()]))
    for date in track(all_dates, description="Cross-sectional normalization"):
        daily = normalize_cross_section(
            feature_df_dict=indexed,
            date=date,
            feature_columns=norm_feature_cols,
            winsorize_pct=float(config.features.winsorize_pct),
        )
        for ticker, row in daily.items():
            for col, val in row.items():
                normalized[ticker].at[pd.Timestamp(date), col] = val

    final_frames = {t: df.reset_index().rename(columns={"index": "Date"}) for t, df in normalized.items()}
    artifacts.feature_frames = final_frames
    save_feature_artifacts(artifacts, config.data.processed_dir, logger=logger)

    n_features = len(feature_cols)
    date_min = min(df["Date"].min() for df in final_frames.values())
    date_max = max(df["Date"].max() for df in final_frames.values())

    nan_rows = []
    for ticker, df in final_frames.items():
        pct_nan = float(df[feature_cols].isna().mean().mean())
        nan_rows.append((ticker, pct_nan))
    avg_nan = float(np.mean([x[1] for x in nan_rows])) if nan_rows else np.nan

    logger.info(
        "features_built tickers=%d feature_count=%d date_range=%s:%s avg_nan_pct=%.4f",
        len(final_frames),
        n_features,
        str(pd.Timestamp(date_min).date()),
        str(pd.Timestamp(date_max).date()),
        avg_nan,
    )
    logger.info("end_timestamp=%s runtime_sec=%.2f", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.time() - t0)


if __name__ == "__main__":
    main()
