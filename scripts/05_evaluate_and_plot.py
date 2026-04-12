"""
Usage: python scripts/05_evaluate_and_plot.py

Standalone evaluation script:
1. Load all fold predictions
2. Recompute metrics
3. Simulate portfolio
4. Generate plots
5. Print rich dashboard table
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_all_metrics
from src.utils.logging_utils import setup_logger
from src.utils.plotting import generate_full_report


def _load_benchmark(config, dates: pd.DatetimeIndex) -> pd.Series:
    idx_path = Path(config.data.raw_dir) / "NIFTY100_INDEX.parquet"
    if not idx_path.exists():
        return pd.Series(np.zeros(len(dates)), index=dates)
    idx = pd.read_parquet(idx_path)
    idx["Date"] = pd.to_datetime(idx["Date"])
    idx = idx.sort_values("Date")
    ret = np.log(idx["Close"] / idx["Close"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.Series(ret.to_numpy(), index=idx["Date"]).reindex(dates).fillna(0.0)


def _dedupe_overlap_predictions(pred_df: pd.DataFrame, policy: str) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df

    if "fold" not in pred_df.columns:
        return pred_df.drop_duplicates(["date", "ticker"], keep="last")

    policy = str(policy).lower()
    if policy in {"none", "keep_all"}:
        return pred_df

    if policy in {"latest", "latest_fold", "last"}:
        return pred_df.sort_values(["date", "ticker", "fold"]).drop_duplicates(["date", "ticker"], keep="last")

    if policy in {"earliest", "first", "first_fold"}:
        return pred_df.sort_values(["date", "ticker", "fold"]).drop_duplicates(["date", "ticker"], keep="first")

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
        out["pred_dir"] = (out["pred_dir"] >= 0.5).astype(int)
        out["actual_dir"] = out["actual_dir"].round().astype(int)
        return out

    return pred_df.sort_values(["date", "ticker", "fold"]).drop_duplicates(["date", "ticker"], keep="last")


def main():
    t0 = time.time()
    config = OmegaConf.load(PROJECT_ROOT / "config" / "config.yaml")
    logger = setup_logger(config)

    metrics_dir = Path(config.logging.metrics_dir)
    pred_files = sorted(metrics_dir.glob("fold_*_predictions.csv"))
    if not pred_files:
        raise FileNotFoundError("No fold prediction CSV files found in results/metrics")

    fold_metrics = []
    all_preds = []
    for pf in pred_files:
        fold = int(pf.stem.split("_")[1])
        df = pd.read_csv(pf)
        df["fold"] = fold
        out = compute_all_metrics(df, config)
        row = dict(out.get("overall", {}))
        row["fold"] = fold
        fold_metrics.append(row)
        all_preds.append(df)

    full_df = pd.concat(all_preds, ignore_index=True)
    overlap_policy = str(getattr(config.evaluation, "aggregate_overlap_policy", "latest_fold"))
    full_eval_df = _dedupe_overlap_predictions(full_df, overlap_policy)
    logger.info(
        "aggregate_overlap policy=%s rows=%d eval_rows=%d",
        overlap_policy,
        len(full_df),
        len(full_eval_df),
    )

    overall_metrics = compute_all_metrics(full_eval_df, config)
    daily = overall_metrics.get("daily", pd.DataFrame())
    bench = _load_benchmark(config, pd.DatetimeIndex(daily.index)) if not daily.empty else pd.Series(dtype=float)

    payload = {
        "fold_metrics": fold_metrics,
        "overall": overall_metrics.get("overall", {}),
        "ic_series": daily["ic"] if "ic" in daily else pd.Series(dtype=float),
        "rank_ic_series": daily["rank_ic"] if "rank_ic" in daily else pd.Series(dtype=float),
        "spread_series": daily["spread"] if "spread" in daily else pd.Series(dtype=float),
        "portfolio_returns": overall_metrics.get("portfolio_returns", pd.Series(dtype=float)),
        "benchmark_returns": bench,
    }
    generate_full_report(payload, config=config, save_dir=config.logging.plots_dir)

    with (metrics_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(overall_metrics.get("overall", {}), fp, indent=2)

    table = Table(title="Evaluation Dashboard")
    table.add_column("Fold", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Sortino", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Ann Return", justify="right")
    table.add_column("IC", justify="right")
    table.add_column("Rank IC", justify="right")
    table.add_column("Spread", justify="right")
    table.add_column("Dir Acc", justify="right")

    for row in sorted(fold_metrics, key=lambda x: x["fold"]):
        table.add_row(
            str(row["fold"]),
            f"{row.get('sharpe_ratio', np.nan):.3f}",
            f"{row.get('sortino_ratio', np.nan):.3f}",
            f"{row.get('max_drawdown', np.nan):.2%}",
            f"{row.get('annualized_return', np.nan):.2%}",
            f"{row.get('ic_mean', np.nan):.4f}",
            f"{row.get('rank_ic_mean', np.nan):.4f}",
            f"{row.get('spread_mean', np.nan):.4f}",
            f"{row.get('directional_accuracy', np.nan):.3f}",
        )

    agg = overall_metrics.get("overall", {})
    table.add_row(
        "ALL",
        f"{agg.get('sharpe_ratio', np.nan):.3f}",
        f"{agg.get('sortino_ratio', np.nan):.3f}",
        f"{agg.get('max_drawdown', np.nan):.2%}",
        f"{agg.get('annualized_return', np.nan):.2%}",
        f"{agg.get('ic_mean', np.nan):.4f}",
        f"{agg.get('rank_ic_mean', np.nan):.4f}",
        f"{agg.get('spread_mean', np.nan):.4f}",
        f"{agg.get('directional_accuracy', np.nan):.3f}",
    )

    Console().print(table)
    logger.info("evaluation_complete folds=%d", len(fold_metrics))
    logger.info("end_timestamp=%s runtime_sec=%.2f", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.time() - t0)


if __name__ == "__main__":
    main()
