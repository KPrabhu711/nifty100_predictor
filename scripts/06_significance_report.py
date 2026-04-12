"""
Usage: python scripts/06_significance_report.py

Post-run statistical significance report using saved prediction outputs.
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
import statsmodels.api as sm
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


def _bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(x, size=x.size, replace=True)
        means.append(float(np.mean(sample)))
    return tuple(np.percentile(means, [2.5, 97.5]).tolist())


def _bootstrap_sharpe_ci(
    returns: np.ndarray, n_boot: int = 2000, seed: int = 42, ann_factor: int = 252
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(returns, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(x, size=x.size, replace=True)
        std = float(np.std(sample, ddof=1))
        mean = float(np.mean(sample))
        vals.append((mean / (std + 1e-8)) * np.sqrt(ann_factor))
    return tuple(np.percentile(vals, [2.5, 97.5]).tolist())


def _hac_mean_stats(values: np.ndarray, max_lags: int = 5) -> dict:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "se_hac": np.nan, "t_hac": np.nan, "pvalue": np.nan}
    model = sm.OLS(x, np.ones((len(x), 1)))
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    return {
        "mean": float(fit.params[0]),
        "se_hac": float(fit.bse[0]),
        "t_hac": float(fit.tvalues[0]),
        "pvalue": float(fit.pvalues[0]),
    }


def _save_bootstrap_plot(values: np.ndarray, title: str, save_path: Path) -> None:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(x, bins=40, color=PRIMARY, alpha=0.85)
    ax.axvline(float(np.mean(x)), color=SECONDARY, linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    config = OmegaConf.load(PROJECT_ROOT / "config" / "config.yaml")
    logger = setup_logger(config)

    pred_path = PROJECT_ROOT / "results" / "metrics" / "all_test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError("results/metrics/all_test_predictions.csv not found")

    df = pd.read_csv(pred_path)
    metrics = compute_all_metrics(df, config)
    daily = metrics["daily"]
    returns = metrics["portfolio_returns"].to_numpy(dtype=np.float64)

    report = {}
    rows = []
    for col, label in [
        ("ic", "IC"),
        ("rank_ic", "Rank IC"),
        ("rank_ic_ret", "Return Rank IC"),
        ("score_rank_ic", "Score Rank IC"),
        ("spread", "Top-Bottom Spread"),
        ("top_k_precision", "Top-K Precision"),
        ("directional_accuracy", "Direction Accuracy"),
    ]:
        vals = daily[col].to_numpy(dtype=np.float64)
        hac = _hac_mean_stats(vals, max_lags=5)
        ci_low, ci_high = _bootstrap_mean_ci(vals, n_boot=2000, seed=42)
        report[col] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            **hac,
        }
        rows.append(
            {
                "metric": label,
                "mean": report[col]["mean"],
                "std": report[col]["std"],
                "ci_low": report[col]["ci_low"],
                "ci_high": report[col]["ci_high"],
                "t_hac": report[col]["t_hac"],
                "pvalue": report[col]["pvalue"],
            }
        )

    sharpe_ci = _bootstrap_sharpe_ci(returns, n_boot=2000, seed=42)
    report["portfolio"] = {
        "sharpe_ratio": float(metrics["overall"]["sharpe_ratio"]),
        "annualized_return": float(metrics["overall"]["annualized_return"]),
        "max_drawdown": float(metrics["overall"]["max_drawdown"]),
        "annualized_turnover": float(metrics["overall"]["annualized_turnover"]),
        "sharpe_ci_low": sharpe_ci[0],
        "sharpe_ci_high": sharpe_ci[1],
    }

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    plots_dir = PROJECT_ROOT / "results" / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(metrics_dir / "significance_table.csv", index=False)
    with (metrics_dir / "significance_report.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    _save_bootstrap_plot(
        daily["ic"].to_numpy(dtype=np.float64),
        "Daily IC Distribution",
        plots_dir / "significance_ic_distribution.png",
    )
    _save_bootstrap_plot(
        returns,
        "Daily Portfolio Return Distribution",
        plots_dir / "significance_portfolio_return_distribution.png",
    )

    table = Table(title="Statistical Significance Report")
    table.add_column("Metric")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("HAC t-stat", justify="right")
    table.add_column("p-value", justify="right")

    for row in rows:
        table.add_row(
            row["metric"],
            f"{row['mean']:.4f}",
            f"{row['std']:.4f}",
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
            f"{row['t_hac']:.3f}",
            f"{row['pvalue']:.4f}",
        )

    portfolio = report["portfolio"]
    table.add_row(
        "Sharpe Ratio",
        f"{portfolio['sharpe_ratio']:.4f}",
        "-",
        f"[{portfolio['sharpe_ci_low']:.4f}, {portfolio['sharpe_ci_high']:.4f}]",
        "-",
        "-",
    )

    Console().print(table)
    logger.info(
        "significance_report_saved path=%s",
        str(metrics_dir / "significance_report.json"),
    )
    logger.info(
        "end_timestamp=%s runtime_sec=%.2f",
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        time.time() - t0,
    )


if __name__ == "__main__":
    main()
