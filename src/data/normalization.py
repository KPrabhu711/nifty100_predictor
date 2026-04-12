from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


EPS = 1e-8


def normalize_cross_section(
    feature_df_dict: dict[str, pd.DataFrame],
    date,
    feature_columns: list[str] | None = None,
    winsorize_pct: float = 0.01,
) -> dict[str, pd.Series]:
    """
    Cross-sectional winsorization + z-score normalization for one date.

    Parameters
    ----------
    feature_df_dict:
        Mapping ticker -> feature DataFrame (must include Date and feature columns).
    date:
        Date key to normalize.
    feature_columns:
        Optional subset of feature columns to normalize. If None, infer all non-target columns.
    winsorize_pct:
        Winsorization percentile for lower/upper clipping.

    Returns
    -------
    dict[ticker, pd.Series]
        Normalized features for the requested date per available ticker.
    """
    date = pd.Timestamp(date)

    aligned_rows: list[pd.Series] = []
    tickers: list[str] = []

    for ticker, df in feature_df_dict.items():
        frame = df
        if "Date" in frame.columns:
            mask = pd.to_datetime(frame["Date"]) == date
            if not mask.any():
                continue
            row = frame.loc[mask].iloc[0]
        else:
            idx = pd.to_datetime(frame.index)
            if date not in idx:
                continue
            row = frame.loc[date]

        if feature_columns is None:
            inferred = [c for c in row.index if c not in {"Date", "ticker"} and not str(c).startswith("target_")]
            feature_columns = inferred

        aligned_rows.append(row[feature_columns])
        tickers.append(ticker)

    if not aligned_rows or feature_columns is None:
        return {}

    mat = np.vstack([r.to_numpy(dtype=np.float64) for r in aligned_rows])

    z = np.zeros_like(mat, dtype=np.float64)
    low_q = winsorize_pct * 100.0
    high_q = (1.0 - winsorize_pct) * 100.0

    for j in range(mat.shape[1]):
        col = mat[:, j]
        valid = np.isfinite(col)
        if valid.sum() == 0:
            z[:, j] = 0.0
            continue

        v = col[valid]
        lo = np.percentile(v, low_q)
        hi = np.percentile(v, high_q)
        clipped = np.clip(col, lo, hi)

        mu = clipped[valid].mean()
        sigma = clipped[valid].std()
        if sigma < EPS:
            z[:, j] = 0.0
        else:
            z[:, j] = (clipped - mu) / (sigma + EPS)

        z[~np.isfinite(z[:, j]), j] = 0.0

    normalized: dict[str, pd.Series] = {}
    for i, ticker in enumerate(tickers):
        row = pd.Series(z[i], index=feature_columns, dtype=np.float64)
        if "log_vol" in row.index:
            row["vol_zscore_cross"] = row["log_vol"]
        normalized[ticker] = row

    return normalized
