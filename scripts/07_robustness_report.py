"""
Usage: python scripts/07_robustness_report.py

Post-run robustness analysis using saved prediction outputs.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_all_metrics
from src.utils.logging_utils import setup_logger


plt.style.use("dark_background")
PRIMARY = "#00D4FF"
SECONDARY = "#FF6B35"
ACCENT = "#7FFF00"


def _dedupe_overlap_predictions(pred_df: pd.DataFrame, policy: str) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df
    if "fold" not in pred_df.columns:
        return pred_df.drop_duplicates(["date", "ticker"], keep="last")

    policy = str(policy).lower()
    if policy in {"none", "keep_all"}:
        return pred_df
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
                "pred_dir_prob_down": "mean",
                "pred_dir_prob_flat": "mean",
                "pred_dir_prob_up": "mean",
                "pred_dir_score": "mean",
                "alpha_score": "mean",
                "actual_return": "mean",
                "actual_dir": "mean",
            }
        )
        out["pred_dir"] = out["pred_dir"].round().astype(int)
        out["actual_dir"] = out["actual_dir"].round().astype(int)
        return out
    return pred_df.sort_values(["date", "ticker", "fold"]).drop_duplicates(
        ["date", "ticker"], keep="last"
    )


def _plot_line(
    df: pd.DataFrame, x_col: str, y_cols: list[str], save_path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PRIMARY, SECONDARY, ACCENT, "#FFD166"]
    for i, col in enumerate(y_cols):
        ax.plot(
            df[x_col],
            df[col],
            marker="o",
            linewidth=2,
            color=colors[i % len(colors)],
            label=col,
        )
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    config = OmegaConf.load(PROJECT_ROOT / "config" / "config.yaml")
    logger = setup_logger(config)

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    plots_dir = PROJECT_ROOT / "results" / "plots"
    pred_path = metrics_dir / "all_test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError("results/metrics/all_test_predictions.csv not found")

    df = pd.read_csv(pred_path)
    df["date"] = pd.to_datetime(df["date"])

    raw_fold_files = sorted(metrics_dir.glob("fold_*_predictions.csv"))
    raw_fold_df = []
    for fp in raw_fold_files:
        fold = int(fp.stem.split("_")[1])
        fdf = pd.read_csv(fp)
        fdf["fold"] = fold
        raw_fold_df.append(fdf)
    raw_fold_df = (
        pd.concat(raw_fold_df, ignore_index=True) if raw_fold_df else pd.DataFrame()
    )

    topk_rows = []
    for score_col in ["alpha_score", "pred_return", "pred_rank"]:
        if score_col not in df.columns:
            continue
        for top_k in [5, 8, 10, 12, 15]:
            cfg = copy.deepcopy(config)
            cfg.evaluation.portfolio_score_col = score_col
            cfg.evaluation.top_k = top_k
            m = compute_all_metrics(df, cfg)["overall"]
            topk_rows.append({"score_col": score_col, "top_k": top_k, **m})
    topk_df = pd.DataFrame(topk_rows)
    topk_df.to_csv(metrics_dir / "robustness_topk.csv", index=False)

    cost_rows = []
    for bps in [0, 5, 10, 20]:
        cfg = copy.deepcopy(config)
        cfg.evaluation.transaction_cost_bps = bps
        m = compute_all_metrics(df, cfg)["overall"]
        cost_rows.append({"transaction_cost_bps": bps, **m})
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(metrics_dir / "robustness_costs.csv", index=False)

    rebalance_rows = []
    for freq in ["D", "W-FRI", "2W", "M"]:
        cfg = copy.deepcopy(config)
        cfg.evaluation.rebalance_frequency = freq
        m = compute_all_metrics(df, cfg)["overall"]
        rebalance_rows.append({"rebalance_frequency": freq, **m})
    rebalance_df = pd.DataFrame(rebalance_rows)
    rebalance_df.to_csv(metrics_dir / "robustness_rebalance.csv", index=False)

    score_rows = []
    for score_col in ["alpha_score", "pred_return", "pred_rank"]:
        if score_col not in df.columns:
            continue
        cfg = copy.deepcopy(config)
        cfg.evaluation.portfolio_score_col = score_col
        m = compute_all_metrics(df, cfg)["overall"]
        score_rows.append({"score_col": score_col, **m})
    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(metrics_dir / "robustness_score_cols.csv", index=False)

    yearly_rows = []
    for year, g in df.groupby(df["date"].dt.year):
        cfg = copy.deepcopy(config)
        m = compute_all_metrics(g, cfg)["overall"]
        yearly_rows.append({"year": int(year), **m})
    yearly_df = pd.DataFrame(yearly_rows)
    yearly_df.to_csv(metrics_dir / "robustness_yearly.csv", index=False)

    overlap_df = pd.DataFrame()
    if not raw_fold_df.empty:
        overlap_rows = []
        for policy in ["latest_fold", "earliest", "mean"]:
            eval_df = _dedupe_overlap_predictions(raw_fold_df, policy)
            cfg = copy.deepcopy(config)
            m = compute_all_metrics(eval_df, cfg)["overall"]
            overlap_rows.append({"overlap_policy": policy, "rows": len(eval_df), **m})
        overlap_df = pd.DataFrame(overlap_rows)
        overlap_df.to_csv(metrics_dir / "robustness_overlap.csv", index=False)

    plots_dir.mkdir(parents=True, exist_ok=True)
    if not topk_df.empty:
        best_topk = topk_df[
            topk_df["score_col"] == config.evaluation.portfolio_score_col
        ].copy()
        if best_topk.empty:
            best_topk = topk_df.copy()
        _plot_line(
            best_topk,
            "top_k",
            ["sharpe_ratio", "annualized_return"],
            plots_dir / "robustness_topk.png",
            "Top-K Sensitivity",
        )
    if not cost_df.empty:
        _plot_line(
            cost_df,
            "transaction_cost_bps",
            ["sharpe_ratio", "annualized_return"],
            plots_dir / "robustness_costs.png",
            "Cost Sensitivity",
        )
    if not yearly_df.empty:
        _plot_line(
            yearly_df,
            "year",
            ["sharpe_ratio", "annualized_return"],
            plots_dir / "robustness_yearly.png",
            "Yearly Performance",
        )

    dashboard = {
        "best_score_col": score_df.sort_values("sharpe_ratio", ascending=False)
        .iloc[0]
        .to_dict()
        if not score_df.empty
        else {},
        "best_top_k": topk_df.sort_values("sharpe_ratio", ascending=False)
        .iloc[0]
        .to_dict()
        if not topk_df.empty
        else {},
        "cost_sensitivity": cost_df.to_dict(orient="records"),
        "rebalance_sensitivity": rebalance_df.to_dict(orient="records"),
        "yearly": yearly_df.to_dict(orient="records"),
    }
    with (metrics_dir / "robustness_dashboard.json").open("w", encoding="utf-8") as fp:
        json.dump(dashboard, fp, indent=2)

    table = Table(title="Robustness Summary")
    table.add_column("Section")
    table.add_column("Best Setting")
    table.add_column("Sharpe", justify="right")
    table.add_column("Ann Return", justify="right")
    table.add_column("Max DD", justify="right")

    if not score_df.empty:
        best = score_df.sort_values("sharpe_ratio", ascending=False).iloc[0]
        table.add_row(
            "Score Column",
            str(best["score_col"]),
            f"{best['sharpe_ratio']:.3f}",
            f"{best['annualized_return']:.2%}",
            f"{best['max_drawdown']:.2%}",
        )
    if not topk_df.empty:
        best = topk_df.sort_values("sharpe_ratio", ascending=False).iloc[0]
        table.add_row(
            "Top-K",
            f"{best['score_col']} / {int(best['top_k'])}",
            f"{best['sharpe_ratio']:.3f}",
            f"{best['annualized_return']:.2%}",
            f"{best['max_drawdown']:.2%}",
        )
    if not cost_df.empty:
        best = cost_df.sort_values("sharpe_ratio", ascending=False).iloc[0]
        table.add_row(
            "Cost",
            f"{int(best['transaction_cost_bps'])} bps",
            f"{best['sharpe_ratio']:.3f}",
            f"{best['annualized_return']:.2%}",
            f"{best['max_drawdown']:.2%}",
        )
    if not rebalance_df.empty:
        best = rebalance_df.sort_values("sharpe_ratio", ascending=False).iloc[0]
        table.add_row(
            "Rebalance",
            str(best["rebalance_frequency"]),
            f"{best['sharpe_ratio']:.3f}",
            f"{best['annualized_return']:.2%}",
            f"{best['max_drawdown']:.2%}",
        )

    Console().print(table)
    logger.info("robustness_reports_saved metrics_dir=%s", str(metrics_dir))
    logger.info(
        "end_timestamp=%s runtime_sec=%.2f",
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        time.time() - t0,
    )


if __name__ == "__main__":
    main()
