"""
Usage: python scripts/08_run_baselines.py

Runs cheap walk-forward baselines using existing processed features.
Baselines:
  - ridge
  - hist_gbm
  - mlp
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import NIFTY100Dataset
from src.training.experiment_utils import (
    add_quantile_direction_labels,
    build_tabular_samples,
    dedupe_overlap_predictions,
    load_full_model_reference,
    metrics_for_predictions,
    save_comparison_plot,
    set_seed,
)
from src.training.walkforward import generate_walkforward_splits
from src.utils.logging_utils import setup_logger


def _model_candidates(seed: int = 42):
    return {
        "ridge": [
            (
                "ridge_a1",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=1.0, random_state=seed)),
                    ]
                ),
            ),
            (
                "ridge_a10",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=10.0, random_state=seed)),
                    ]
                ),
            ),
            (
                "ridge_a100",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=100.0, random_state=seed)),
                    ]
                ),
            ),
        ],
        "hist_gbm": [
            (
                "hgbm_d3",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=3,
                    max_iter=200,
                    min_samples_leaf=50,
                    random_state=seed,
                ),
            ),
            (
                "hgbm_d5",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=5,
                    max_iter=250,
                    min_samples_leaf=50,
                    random_state=seed,
                ),
            ),
        ],
        "mlp": [
            (
                "mlp_128_64",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            MLPRegressor(
                                hidden_layer_sizes=(128, 64),
                                activation="relu",
                                solver="adam",
                                alpha=1e-4,
                                batch_size=1024,
                                learning_rate_init=1e-3,
                                max_iter=60,
                                early_stopping=True,
                                validation_fraction=0.1,
                                n_iter_no_change=8,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "mlp_256_128",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            MLPRegressor(
                                hidden_layer_sizes=(256, 128),
                                activation="relu",
                                solver="adam",
                                alpha=5e-4,
                                batch_size=1024,
                                learning_rate_init=8e-4,
                                max_iter=60,
                                early_stopping=True,
                                validation_fraction=0.1,
                                n_iter_no_change=8,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            ),
        ],
    }


def _fit_and_predict(model, x_train, y_train, x_pred):
    model.fit(x_train, y_train)
    pred = model.predict(x_pred)
    return np.asarray(pred, dtype=np.float32)


def _prediction_frame(meta_df: pd.DataFrame, pred_return: np.ndarray) -> pd.DataFrame:
    out = meta_df.copy()
    out["pred_return"] = np.asarray(pred_return, dtype=np.float32)
    out["pred_rank"] = out["pred_return"]
    out = add_quantile_direction_labels(out, score_col="pred_return", q=0.30)
    return out[
        [
            "date",
            "ticker",
            "pred_return",
            "pred_rank",
            "pred_dir",
            "actual_return",
            "actual_dir",
        ]
    ]


def main() -> None:
    t0 = time.time()
    set_seed(42)
    config = OmegaConf.load(PROJECT_ROOT / "config" / "config.yaml")
    logger = setup_logger(config)

    dataset = NIFTY100Dataset(config)
    all_dates = [dataset.calendar[i].strftime("%Y-%m-%d") for i in dataset.valid_idxs]
    splits = generate_walkforward_splits(all_dates, config)
    if not splits:
        raise RuntimeError("No walk-forward splits generated")

    baseline_dir = PROJECT_ROOT / "results" / "metrics" / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    candidates = _model_candidates(seed=42)
    for baseline_name, model_list in candidates.items():
        logger.info("baseline_start name=%s", baseline_name)
        all_preds = []
        fold_rows = []

        for split in splits:
            fold = split["fold"]
            x_train, meta_train, _ = build_tabular_samples(
                dataset, split["train_dates"]
            )
            x_val, meta_val, _ = build_tabular_samples(dataset, split["val_dates"])
            x_test, meta_test, _ = build_tabular_samples(dataset, split["test_dates"])

            y_train = meta_train["actual_return"].to_numpy(dtype=np.float32)
            y_val = meta_val["actual_return"].to_numpy(dtype=np.float32)
            y_trainval = np.concatenate([y_train, y_val], axis=0)
            x_trainval = np.concatenate([x_train, x_val], axis=0)

            best = None
            for candidate_name, model in model_list:
                val_pred = _fit_and_predict(model, x_train, y_train, x_val)
                val_df = _prediction_frame(meta_val, val_pred)
                val_metrics = metrics_for_predictions(
                    val_df, config, score_col="pred_return"
                )["overall"]
                key = (
                    float(val_metrics.get("score_rank_ic_mean", -np.inf)),
                    float(val_metrics.get("sharpe_ratio", -np.inf)),
                    float(val_metrics.get("annualized_return", -np.inf)),
                )
                if best is None or key > best[0]:
                    best = (key, candidate_name, model, val_metrics)

            assert best is not None
            _, candidate_name, best_model, best_val_metrics = best
            test_pred = _fit_and_predict(best_model, x_trainval, y_trainval, x_test)
            test_df = _prediction_frame(meta_test, test_pred)
            test_df["fold"] = fold
            all_preds.append(test_df)

            fold_metrics = metrics_for_predictions(
                test_df, config, score_col="pred_return"
            )["overall"]
            fold_metrics["fold"] = fold
            fold_metrics["baseline"] = baseline_name
            fold_metrics["candidate"] = candidate_name
            fold_rows.append(fold_metrics)

            with (baseline_dir / f"{baseline_name}_fold_{fold}_metrics.json").open(
                "w", encoding="utf-8"
            ) as fp:
                json.dump({"val": best_val_metrics, "test": fold_metrics}, fp, indent=2)
            test_df.to_csv(
                baseline_dir / f"{baseline_name}_fold_{fold}_predictions.csv",
                index=False,
            )

        baseline_pred = pd.concat(all_preds, ignore_index=True)
        baseline_pred = dedupe_overlap_predictions(
            baseline_pred, str(config.evaluation.aggregate_overlap_policy)
        )
        baseline_pred.to_csv(
            baseline_dir / f"{baseline_name}_all_predictions.csv", index=False
        )
        overall = metrics_for_predictions(
            baseline_pred, config, score_col="pred_return"
        )["overall"]
        overall["variant"] = baseline_name
        summary_rows.append(overall)
        pd.DataFrame(fold_rows).to_csv(
            baseline_dir / f"{baseline_name}_fold_summary.csv", index=False
        )
        logger.info(
            "baseline_complete name=%s sharpe=%.4f ann_return=%.4f rank_ic=%.4f",
            baseline_name,
            float(overall.get("sharpe_ratio", np.nan)),
            float(overall.get("annualized_return", np.nan)),
            float(overall.get("rank_ic_mean", np.nan)),
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("sharpe_ratio", ascending=False)
    full_ref = load_full_model_reference(
        PROJECT_ROOT / "results" / "metrics" / "aggregate_metrics.json"
    )
    if full_ref is not None:
        summary_df = pd.concat(
            [pd.DataFrame([full_ref]), summary_df], ignore_index=True
        )

    summary_df.to_csv(
        PROJECT_ROOT / "results" / "metrics" / "baselines_summary.csv", index=False
    )
    save_comparison_plot(
        summary_df[
            [
                c
                for c in [
                    "variant",
                    "sharpe_ratio",
                    "annualized_return",
                    "rank_ic_mean",
                ]
                if c in summary_df.columns
            ]
        ],
        PROJECT_ROOT / "results" / "plots" / "baselines_comparison.png",
        title="Cheap Baseline Comparison",
    )

    table = Table(title="Cheap Baselines")
    table.add_column("Variant")
    table.add_column("Sharpe", justify="right")
    table.add_column("Ann Return", justify="right")
    table.add_column("Rank IC", justify="right")
    table.add_column("Top-K Prec", justify="right")

    for _, row in summary_df.iterrows():
        table.add_row(
            str(row.get("variant", "-")),
            f"{row.get('sharpe_ratio', np.nan):.3f}",
            f"{row.get('annualized_return', np.nan):.2%}",
            f"{row.get('rank_ic_mean', np.nan):.4f}",
            f"{row.get('top_k_precision', np.nan):.3f}",
        )
    Console().print(table)
    logger.info(
        "baselines_saved path=%s",
        str(PROJECT_ROOT / "results" / "metrics" / "baselines_summary.csv"),
    )
    logger.info(
        "end_timestamp=%s runtime_sec=%.2f",
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        time.time() - t0,
    )


if __name__ == "__main__":
    main()
