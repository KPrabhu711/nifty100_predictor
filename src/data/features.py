from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


EPS = 1e-8


def _direction_labels(target_raw: pd.Series, mode: str, threshold: float) -> pd.Series:
    mode = str(mode).lower()
    if mode in {"binary", "sign"}:
        out = (target_raw > 0.0).astype(float)
        out[target_raw.isna()] = np.nan
        return out

    if mode in {"ternary", "ternary_threshold", "3class"}:
        arr = np.where(target_raw > threshold, 2.0, np.where(target_raw < -threshold, 0.0, 1.0))
        out = pd.Series(arr, index=target_raw.index, dtype=float)
        out[target_raw.isna()] = np.nan
        return out

    if mode in {"ternary_quantile", "ternary_quantile_cs", "ternary_quantile_cross_section"}:
        # Filled later in build_feature_artifacts using cross-sectional quantiles per date.
        return pd.Series(np.nan, index=target_raw.index, dtype=float)

    raise ValueError(f"Unsupported direction_label_mode: {mode}")


def _apply_cross_sectional_direction_labels(feature_frames: dict[str, pd.DataFrame], config, logger=None) -> None:
    mode = str(getattr(config.features, "direction_label_mode", "binary")).lower()
    if mode not in {"ternary_quantile", "ternary_quantile_cs", "ternary_quantile_cross_section"}:
        return

    q = float(getattr(config.features, "direction_quantile", 0.3))
    q = float(np.clip(q, 0.05, 0.45))
    source = str(getattr(config.features, "direction_target_source", "raw")).lower()
    horizons = list(config.features.target_horizons)

    for h in horizons:
        target_col = f"target_res_{h}" if source in {"res", "residual", "resid"} else f"target_raw_{h}"
        dir_col = f"target_dir_{h}"

        wide_dict: dict[str, pd.Series] = {}
        for ticker, frame in feature_frames.items():
            s = frame[["Date", target_col]].copy()
            s["Date"] = pd.to_datetime(s["Date"])
            wide_dict[ticker] = s.set_index("Date")[target_col]

        wide = pd.DataFrame(wide_dict).sort_index()
        if wide.empty:
            continue

        lo = wide.quantile(q, axis=1)
        hi = wide.quantile(1.0 - q, axis=1)

        arr = wide.to_numpy(dtype=np.float64)
        lo_arr = lo.to_numpy(dtype=np.float64)[:, None]
        hi_arr = hi.to_numpy(dtype=np.float64)[:, None]
        labels = np.where(np.isnan(arr), np.nan, np.where(arr <= lo_arr, 0.0, np.where(arr >= hi_arr, 2.0, 1.0)))
        label_wide = pd.DataFrame(labels, index=wide.index, columns=wide.columns)

        for ticker, frame in feature_frames.items():
            idx = pd.to_datetime(frame["Date"])
            frame[dir_col] = label_wide[ticker].reindex(idx).to_numpy(dtype=np.float64)

    if logger:
        logger.info("direction_labels_cross_sectional_applied mode=%s source=%s quantile=%.2f", mode, source, q)


@dataclass
class FeatureBuildArtifacts:
    feature_frames: dict[str, pd.DataFrame]
    regime_features: pd.DataFrame
    sector_map: pd.DataFrame
    feature_columns: list[str]


def _clip_returns(series: pd.Series, min_val: float = -0.5, max_val: float = 0.5) -> pd.Series:
    return series.clip(lower=min_val, upper=max_val)


def _safe_log_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    ratio = (num / den).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.clip(lower=EPS)
    return np.log(ratio)


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100.0 - (100.0 / (1.0 + rs))


def _rolling_slope(log_close: pd.Series, window: int = 5) -> pd.Series:
    x = np.arange(window, dtype=np.float64)

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        try:
            return float(np.polyfit(x, values, 1)[0])
        except Exception:
            return np.nan

    return log_close.rolling(window).apply(lambda arr: _slope(arr), raw=True)


def _prepare_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").drop_duplicates(subset=["Date"])
    out = out.set_index("Date")
    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    return out[cols]


def _compute_regime_features(index_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    idx = _prepare_price_frame(index_df)
    vix = _prepare_price_frame(vix_df)
    regime = pd.DataFrame(index=idx.index)

    nifty_log_ret = _clip_returns(_safe_log_ratio(idx["Close"], idx["Close"].shift(1)))
    vix_level = vix["Close"].reindex(idx.index)
    regime["vix_level"] = vix_level
    regime["vix_chg"] = vix_level.pct_change()
    regime["vix_zscore"] = (vix_level - vix_level.rolling(60).mean()) / (vix_level.rolling(60).std() + EPS)
    regime["mkt_ret_5"] = nifty_log_ret.rolling(5).sum()
    regime["mkt_vol_20"] = nifty_log_ret.rolling(20).std()
    regime["mkt_trend"] = np.sign(idx["Close"] - idx["Close"].rolling(60).mean())
    regime = regime.reset_index().rename(columns={"index": "Date"})
    return regime


def compute_features(
    ticker_df: pd.DataFrame,
    index_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    sector_peer_ret: pd.Series,
    rs_pct: pd.Series,
    config,
) -> pd.DataFrame:
    df = _prepare_price_frame(ticker_df)
    idx = _prepare_price_frame(index_df)

    out = pd.DataFrame(index=df.index)

    out["log_ret"] = _clip_returns(_safe_log_ratio(df["Close"], df["Close"].shift(1)))
    out["overnight_ret"] = _clip_returns(_safe_log_ratio(df["Open"], df["Close"].shift(1)))
    out["intraday_ret"] = _clip_returns(_safe_log_ratio(df["Close"], df["Open"]))
    out["hl_range"] = (df["High"] - df["Low"]) / (df["Close"] + EPS)
    out["co_spread"] = (df["Close"] - df["Open"]) / (df["Open"] + EPS)

    for w in config.features.momentum_windows:
        out[f"mom_{w}"] = _clip_returns(_safe_log_ratio(df["Close"], df["Close"].shift(w)))

    out["drawdown"] = df["Close"] / (df["Close"].rolling(20).max() + EPS) - 1.0

    for w in config.features.vol_windows:
        out[f"rvol_{w}"] = out["log_ret"].rolling(w).std()

    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(14).mean() / (df["Close"] + EPS)
    out["vol_zscore"] = (out["rvol_20"] - out["rvol_20"].rolling(60).mean()) / (out["rvol_20"].rolling(60).std() + EPS)

    out["log_vol"] = np.log(df["Volume"].clip(lower=0.0) + 1.0)
    out["vol_chg"] = out["log_vol"].diff()
    out["vol_surprise"] = out["log_vol"] - out["log_vol"].rolling(20).mean()
    out["vol_zscore_cross"] = np.nan

    nifty_log_ret = _clip_returns(_safe_log_ratio(idx["Close"], idx["Close"].shift(1)))
    nifty_log_ret = nifty_log_ret.reindex(out.index)
    out["ret_vs_mkt"] = out["log_ret"] - nifty_log_ret
    out["ret_vs_sector"] = out["log_ret"] - sector_peer_ret.reindex(out.index)
    out["rs_pct"] = rs_pct.reindex(out.index)

    out["ma_dist_20"] = df["Close"] / (df["Close"].rolling(20).mean() + EPS) - 1.0
    out["ma_dist_60"] = df["Close"] / (df["Close"].rolling(60).mean() + EPS) - 1.0
    out["rsi_14"] = _compute_rsi(df["Close"], window=14)

    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    out["bb_dist"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower + EPS)

    log_close = np.log(df["Close"].clip(lower=EPS))
    out["slope_5"] = _rolling_slope(log_close, window=5)

    target_horizons = list(config.features.target_horizons)
    idx_close = idx["Close"].reindex(out.index)
    direction_mode = str(getattr(config.features, "direction_label_mode", "binary"))
    direction_threshold = float(getattr(config.features, "direction_threshold", 0.0))
    direction_source = str(getattr(config.features, "direction_target_source", "raw")).lower()
    for h in target_horizons:
        target_raw = _clip_returns(_safe_log_ratio(df["Close"].shift(-h), df["Close"]))
        idx_h = _clip_returns(_safe_log_ratio(idx_close.shift(-h), idx_close))
        target_res = target_raw - idx_h
        out[f"target_raw_{h}"] = target_raw
        out[f"target_res_{h}"] = target_res

        direction_base = target_res if direction_source in {"res", "residual", "resid"} else target_raw
        out[f"target_dir_{h}"] = _direction_labels(
            target_raw=direction_base,
            mode=direction_mode,
            threshold=direction_threshold,
        )

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.reset_index().rename(columns={"index": "Date"})
    return out


def build_feature_artifacts(config, logger=None) -> FeatureBuildArtifacts:
    raw_dir = Path(config.data.raw_dir)
    universe = pd.read_csv(config.data.universe_file)
    tickers = universe["ticker"].astype(str).tolist()
    expected_n = int(config.universe.n_stocks)
    if logger and len(tickers) != expected_n:
        logger.warning("universe_size_mismatch config_n=%d csv_n=%d", expected_n, len(tickers))

    index_path = raw_dir / "NIFTY100_INDEX.parquet"
    vix_path = raw_dir / "INDIA_VIX.parquet"
    if not index_path.exists() or not vix_path.exists():
        raise FileNotFoundError("Missing NIFTY100_INDEX.parquet or INDIA_VIX.parquet in data/raw")

    index_df = pd.read_parquet(index_path)
    vix_df = pd.read_parquet(vix_path)

    raw_ticker_data: dict[str, pd.DataFrame] = {}
    log_ret_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = raw_dir / f"{ticker}.parquet"
        if not path.exists():
            if logger:
                logger.warning("missing_raw_ticker ticker=%s", ticker)
            continue
        tdf = pd.read_parquet(path)
        try:
            prep = _prepare_price_frame(tdf)
        except Exception as exc:
            if logger:
                logger.warning("bad_raw_ticker ticker=%s err=%s", ticker, str(exc))
            continue

        raw_ticker_data[ticker] = tdf
        log_ret_map[ticker] = _clip_returns(_safe_log_ratio(prep["Close"], prep["Close"].shift(1)))

    returns_wide = pd.DataFrame(log_ret_map).sort_index()
    rs_pct_wide = returns_wide.rank(axis=1, pct=True)

    sector_map = universe[["ticker", "sector"]].copy()
    sector_map = sector_map[sector_map["ticker"].isin(raw_ticker_data.keys())]

    sector_peers: dict[str, pd.Series] = {}
    for sector, sdf in sector_map.groupby("sector"):
        sector_tickers = sdf["ticker"].tolist()
        sec_ret = returns_wide[sector_tickers]
        sec_sum = sec_ret.sum(axis=1)
        sec_count = sec_ret.notna().sum(axis=1)
        for ticker in sector_tickers:
            denom = (sec_count - 1).replace(0, np.nan)
            peer = (sec_sum - sec_ret[ticker]) / denom
            sector_peers[ticker] = peer

    feature_frames: dict[str, pd.DataFrame] = {}
    for ticker, tdf in raw_ticker_data.items():
        frame = compute_features(
            ticker_df=tdf,
            index_df=index_df,
            vix_df=vix_df,
            sector_peer_ret=sector_peers.get(ticker, pd.Series(dtype=float)),
            rs_pct=rs_pct_wide.get(ticker, pd.Series(dtype=float)),
            config=config,
        )
        frame["ticker"] = ticker
        feature_frames[ticker] = frame

    _apply_cross_sectional_direction_labels(feature_frames=feature_frames, config=config, logger=logger)

    if not feature_frames:
        raise RuntimeError("No feature frames created. Check raw data availability.")

    sample_cols = list(next(iter(feature_frames.values())).columns)
    feature_columns = [
        c
        for c in sample_cols
        if c not in {"Date", "ticker"} and not c.startswith("target_")
    ]

    regime_df = _compute_regime_features(index_df=index_df, vix_df=vix_df)

    if logger:
        logger.info(
            "feature_artifacts_built tickers=%d feature_cols=%d dates=%d",
            len(feature_frames),
            len(feature_columns),
            len(regime_df),
        )

    return FeatureBuildArtifacts(
        feature_frames=feature_frames,
        regime_features=regime_df,
        sector_map=sector_map.reset_index(drop=True),
        feature_columns=feature_columns,
    )


def save_feature_artifacts(
    artifacts: FeatureBuildArtifacts,
    processed_dir: str | Path,
    logger=None,
) -> None:
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ticker, frame in artifacts.feature_frames.items():
        frame.sort_values("Date").to_parquet(out_dir / f"{ticker}_features.parquet", index=False)

    artifacts.regime_features.sort_values("Date").to_parquet(out_dir / "regime_features.parquet", index=False)
    artifacts.sector_map.to_parquet(out_dir / "sector_map.parquet", index=False)

    with (out_dir / "feature_columns.json").open("w", encoding="utf-8") as fp:
        json.dump(artifacts.feature_columns, fp, indent=2)

    if logger:
        logger.info(
            "feature_artifacts_saved ticker_files=%d processed_dir=%s",
            len(artifacts.feature_frames),
            str(out_dir),
        )
