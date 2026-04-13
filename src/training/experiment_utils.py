from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataset import NIFTY100Dataset
from src.evaluation.metrics import compute_all_metrics


plt.style.use("dark_background")
PRIMARY = "#00D4FF"
SECONDARY = "#FF6B35"
ACCENT = "#7FFF00"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dedupe_overlap_predictions(
    pred_df: pd.DataFrame, policy: str = "latest_fold"
) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df
    if "fold" not in pred_df.columns:
        return pred_df.drop_duplicates(["date", "ticker"], keep="last")

    policy = str(policy).lower()
    if policy in {"latest", "latest_fold", "last"}:
        return pred_df.sort_values(["date", "ticker", "fold"]).drop_duplicates(
            ["date", "ticker"], keep="last"
        )
    if policy in {"earliest", "first", "first_fold"}:
        return pred_df.sort_values(["date", "ticker", "fold"]).drop_duplicates(
            ["date", "ticker"], keep="first"
        )
    if policy in {"mean", "average", "avg"}:
        grp = pred_df.groupby(["date", "ticker"], as_index=False)
        out = grp.agg(
            {
                "pred_return": "mean",
                "pred_rank": "mean",
                "pred_dir": "mean",
                "actual_return": "mean",
                "actual_dir": "mean",
            }
        )
        out["pred_dir"] = out["pred_dir"].round().astype(int)
        out["actual_dir"] = out["actual_dir"].round().astype(int)
        return out
    return pred_df.drop_duplicates(["date", "ticker"], keep="last")


def build_tabular_feature_names(
    feature_columns: list[str], regime_columns: list[str]
) -> list[str]:
    names = []
    names.extend([f"latest__{c}" for c in feature_columns])
    for window in [5, 20, 60]:
        names.extend([f"mean{window}__{c}" for c in feature_columns])
        names.extend([f"std{window}__{c}" for c in feature_columns])
    names.extend([f"regime__{c}" for c in regime_columns])
    names.extend(
        [
            "ret_sum_5",
            "ret_std_5",
            "ret_sum_20",
            "ret_std_20",
            "ret_sum_40",
            "ret_std_40",
            "ret_last",
            "ret_mean_40",
            "ret_skew_40",
        ]
    )
    return names


def _safe_stats(arr: np.ndarray) -> tuple[float, float]:
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _skew(arr: np.ndarray) -> float:
    if arr.size < 3:
        return 0.0
    x = arr.astype(np.float64)
    mu = float(x.mean())
    sd = float(x.std())
    if sd < 1e-8:
        return 0.0
    return float(np.mean(((x - mu) / (sd + 1e-8)) ** 3))


def tabular_feature_vector(
    window: np.ndarray, raw_returns: np.ndarray, regime: np.ndarray
) -> np.ndarray:
    feats = []
    feats.append(window[-1])
    for size in [5, 20, 60]:
        w = window[-min(size, len(window)) :]
        feats.append(np.mean(w, axis=0))
        feats.append(np.std(w, axis=0))

    feats.append(regime.astype(np.float32))

    rr = raw_returns.astype(np.float32)
    r5 = rr[-min(5, len(rr)) :]
    r20 = rr[-min(20, len(rr)) :]
    r40 = rr[-min(40, len(rr)) :]
    stats = np.array(
        [
            float(np.sum(r5)),
            float(np.std(r5)),
            float(np.sum(r20)),
            float(np.std(r20)),
            float(np.sum(r40)),
            float(np.std(r40)),
            float(rr[-1]) if rr.size else 0.0,
            float(np.mean(r40)) if r40.size else 0.0,
            _skew(r40),
        ],
        dtype=np.float32,
    )
    feats.append(stats)

    out = np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in feats])
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_tabular_samples(
    dataset: NIFTY100Dataset, date_spec
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    if isinstance(date_spec, tuple) and len(date_spec) == 2:
        date_set = None
        start, end = pd.Timestamp(date_spec[0]), pd.Timestamp(date_spec[1])
    else:
        date_set = {pd.Timestamp(d) for d in date_spec}
        start, end = None, None

    X = []
    rows = []
    feature_names = build_tabular_feature_names(
        dataset.feature_columns, dataset.regime_cols
    )

    for t_idx in dataset.valid_idxs:
        date = dataset.calendar[t_idx]
        if date_set is not None:
            if date not in date_set:
                continue
        else:
            assert start is not None and end is not None
            if not (start <= date <= end):
                continue

        regime = dataset.regime_arr[t_idx]
        for i, ticker in enumerate(dataset.tickers):
            if dataset.target_mask[i, t_idx] <= 0.5:
                continue

            window = dataset.features_arr[
                i, t_idx - dataset.lookback + 1 : t_idx + 1, :
            ]
            raw_returns = dataset.log_rets[
                i, t_idx - dataset.corr_window + 1 : t_idx + 1
            ]
            X.append(tabular_feature_vector(window, raw_returns, regime))
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "actual_return": float(dataset.targets_reg[i, t_idx]),
                    "actual_dir": int(dataset.targets_dir[i, t_idx]),
                }
            )

    return np.asarray(X, dtype=np.float32), pd.DataFrame(rows), feature_names


def add_quantile_direction_labels(
    pred_df: pd.DataFrame, score_col: str = "pred_return", q: float = 0.30
) -> pd.DataFrame:
    df = pred_df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    out = []
    for _, g in df.groupby("date", sort=False):
        lo = float(g[score_col].quantile(q))
        hi = float(g[score_col].quantile(1.0 - q))
        pred_dir = np.where(g[score_col] <= lo, 0, np.where(g[score_col] >= hi, 2, 1))
        gg = g.copy()
        gg["pred_dir"] = pred_dir.astype(int)
        out.append(gg)
    return pd.concat(out, ignore_index=True) if out else df


def metrics_for_predictions(pred_df: pd.DataFrame, config, score_col: str) -> dict:
    cfg = copy.deepcopy(config)
    cfg.evaluation.portfolio_score_col = score_col
    return compute_all_metrics(pred_df, cfg)


def save_comparison_plot(
    df: pd.DataFrame, save_path: Path, title: str, x_col: str = "variant"
) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["sharpe_ratio", "annualized_return", "rank_ic_mean"]
    colors = [PRIMARY, SECONDARY, ACCENT]
    for ax, metric, color in zip(axes, metrics, colors):
        ax.bar(df[x_col].astype(str), df[metric], color=color, alpha=0.85)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_full_model_reference(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        data["variant"] = "full_saved"
        return data
    except Exception:
        return None
