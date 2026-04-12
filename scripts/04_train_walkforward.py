"""
Usage: python scripts/04_train_walkforward.py

1. Load config and pretrained backbone weights
2. Generate walk-forward splits
3. Train/evaluate each fold
4. Save fold metrics, predictions, and plots
5. Aggregate and print final summary table
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import NIFTY100Dataset, build_dataloaders
from src.evaluation.metrics import compute_all_metrics
from src.models.full_model import NIFTY100PredictionModel
from src.models.graph import build_combined_graph
from src.training.trainer import Trainer
from src.training.walkforward import generate_walkforward_splits
from src.utils.logging_utils import setup_logger
from src.utils.plotting import (
    generate_full_report,
    plot_ic_over_time,
    plot_portfolio_cumulative_returns,
    plot_top_bottom_spread,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_benchmark_returns(config, dates: pd.DatetimeIndex) -> pd.Series:
    idx_path = Path(config.data.raw_dir) / "NIFTY100_INDEX.parquet"
    if not idx_path.exists():
        return pd.Series(np.zeros(len(dates)), index=dates)
    idx = pd.read_parquet(idx_path)
    idx["Date"] = pd.to_datetime(idx["Date"])
    idx = idx.sort_values("Date")
    bench = np.log(idx["Close"] / idx["Close"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s = pd.Series(bench.to_numpy(), index=idx["Date"])
    return s.reindex(dates).fillna(0.0)


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
    set_seed(42)
    config = OmegaConf.load(PROJECT_ROOT / "config" / "config.yaml")
    logger = setup_logger(config)

    base_ds = NIFTY100Dataset(config)
    all_dates = [base_ds.calendar[i].strftime("%Y-%m-%d") for i in base_ds.valid_idxs]
    logger.info(
        "dataset_shapes n_stocks=%d lookback=%d feature_dim=%d valid_dates=%d",
        base_ds.n_stocks,
        int(config.features.lookback),
        len(base_ds.feature_columns),
        len(all_dates),
    )

    splits = generate_walkforward_splits(all_dates, config)
    if not splits:
        raise RuntimeError("No walk-forward splits generated")

    pretrain_ckpt = Path(config.logging.checkpoint_dir) / "pretrain_best.pt"
    fold_metrics = []
    combined_preds = []

    for split in splits:
        fold = split["fold"]
        logger.info("fold_start fold=%d", fold)

        config.runtime_split_dates = {
            "train": split["train_dates"],
            "val": split["val_dates"],
            "test": split["test_dates"],
        }
        train_dl = build_dataloaders(config, "train")
        val_dl = build_dataloaders(config, "val")
        test_dl = build_dataloaders(config, "test")

        if len(train_dl.dataset) == 0 or len(val_dl.dataset) == 0 or len(test_dl.dataset) == 0:
            logger.warning("skip_fold_empty_dataset fold=%d train=%d val=%d test=%d", fold, len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset))
            continue

        model = NIFTY100PredictionModel(config=config, feature_dim=len(train_dl.dataset.feature_columns))

        sample_batch = next(iter(train_dl))
        sid0 = sample_batch["sector_ids"][0]
        rr0 = sample_batch["raw_returns"][0]
        dummy_emb = torch.randn(sid0.numel(), int(config.model.embedding_dim))
        graph_probe = build_combined_graph(sid0, dummy_emb, rr0, config)
        sec_e = graph_probe["sector"][0].shape[1]
        corr_e = graph_probe["corr"][0].shape[1]
        emb_e = graph_probe["emb_sim"][0].shape[1]
        logger.info("graph_edge_counts fold=%d sector=%d corr=%d emb_sim=%d", fold, sec_e, corr_e, emb_e)

        if pretrain_ckpt.exists():
            payload = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)
            miss, unexp = model.temporal_encoder.load_state_dict(payload["backbone_state_dict"], strict=False)
            logger.info("loaded_pretrain fold=%d missing=%d unexpected=%d", fold, len(miss), len(unexp))

        trainer = Trainer(model=model, config=config, logger=logger)
        ckpt_path = Path(config.logging.checkpoint_dir) / f"fold_{fold}_best.pt"
        checkpoint_ready = False
        if ckpt_path.exists():
            logger.info("resume_existing_checkpoint fold=%d path=%s", fold, str(ckpt_path))
            try:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                trainer.model.load_state_dict(state["model_state_dict"], strict=True)
                checkpoint_ready = True
            except Exception as exc:
                logger.warning("checkpoint_incompatible_retrain fold=%d path=%s err=%s", fold, str(ckpt_path), str(exc))

        if not checkpoint_ready:
            trainer.train_fold(train_dl=train_dl, val_dl=val_dl, fold_id=fold)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            trainer.model.load_state_dict(state["model_state_dict"], strict=True)

        val_pred_df, val_metrics = trainer.infer(val_dl)
        alpha_selection = val_metrics.get("alpha_selection", {}) if isinstance(val_metrics, dict) else {}
        if alpha_selection:
            logger.info(
                "alpha_selection fold=%d weights=%s score_col=%s val_sharpe=%.4f val_score_rank_ic=%.4f",
                fold,
                alpha_selection.get("weights"),
                alpha_selection.get("score_col"),
                float(alpha_selection.get("metrics", {}).get("sharpe_ratio", np.nan)),
                float(alpha_selection.get("metrics", {}).get("score_rank_ic_mean", np.nan)),
            )

        pred_df, test_metrics = trainer.infer(test_dl, alpha_selection=alpha_selection)
        if pred_df.empty:
            logger.warning("empty_test_predictions fold=%d", fold)
            continue

        pred_df["fold"] = fold

        pred_path = Path(config.logging.metrics_dir) / f"fold_{fold}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        metric_obj = test_metrics if test_metrics else compute_all_metrics(pred_df, config)
        overall = metric_obj.get("overall", {})
        overall["fold"] = fold
        if alpha_selection:
            weights = list(alpha_selection.get("weights", []))
            if len(weights) == 3:
                overall["alpha_w_ret"] = weights[0]
                overall["alpha_w_rank"] = weights[1]
                overall["alpha_w_dir"] = weights[2]
            overall["alpha_score_col"] = alpha_selection.get("score_col", "alpha_score")
        fold_metrics.append(overall)
        combined_preds.append(pred_df)

        daily = metric_obj.get("daily", pd.DataFrame())
        if not daily.empty:
            plot_ic_over_time(
                daily["ic"],
                daily["rank_ic"],
                Path(config.logging.plots_dir) / f"fold_{fold}_ic_over_time.png",
            )
            plot_top_bottom_spread(
                daily["spread"],
                Path(config.logging.plots_dir) / f"fold_{fold}_spread.png",
            )

            bench = _load_benchmark_returns(config, daily.index)
            plot_portfolio_cumulative_returns(
                metric_obj["portfolio_returns"],
                bench,
                Path(config.logging.plots_dir) / f"fold_{fold}_cumulative_returns.png",
            )

        with (Path(config.logging.metrics_dir) / f"fold_{fold}_metrics.json").open("w", encoding="utf-8") as fp:
            json.dump({"overall": overall, "alpha_selection": alpha_selection}, fp, indent=2)

        logger.log({f"fold_{fold}_rank_ic": overall.get("rank_ic_mean", np.nan), f"fold_{fold}_sharpe": overall.get("sharpe_ratio", np.nan)})

    if not combined_preds:
        raise RuntimeError("No fold predictions were generated")

    all_preds = pd.concat(combined_preds, ignore_index=True)
    unique_pairs = all_preds[["date", "ticker"]].drop_duplicates().shape[0]
    duplicate_pairs = len(all_preds) - unique_pairs

    overlap_policy = str(getattr(config.evaluation, "aggregate_overlap_policy", "latest_fold"))
    all_preds_eval = _dedupe_overlap_predictions(all_preds, overlap_policy)
    logger.info(
        "aggregate_overlap policy=%s rows=%d unique_pairs=%d duplicates=%d eval_rows=%d",
        overlap_policy,
        len(all_preds),
        unique_pairs,
        duplicate_pairs,
        len(all_preds_eval),
    )

    all_metrics = compute_all_metrics(all_preds_eval, config)
    daily = all_metrics.get("daily", pd.DataFrame())

    benchmark = _load_benchmark_returns(config, pd.DatetimeIndex(daily.index)) if not daily.empty else pd.Series(dtype=float)

    report_payload = {
        "fold_metrics": fold_metrics,
        "overall": all_metrics.get("overall", {}),
        "ic_series": daily["ic"] if "ic" in daily else pd.Series(dtype=float),
        "rank_ic_series": daily["rank_ic"] if "rank_ic" in daily else pd.Series(dtype=float),
        "spread_series": daily["spread"] if "spread" in daily else pd.Series(dtype=float),
        "portfolio_returns": all_metrics.get("portfolio_returns", pd.Series(dtype=float)),
        "benchmark_returns": benchmark,
        "tickers": base_ds.tickers,
    }
    generate_full_report(report_payload, config=config, save_dir=config.logging.plots_dir)

    summary_df = pd.DataFrame(fold_metrics)
    summary_df.to_csv(Path(config.logging.metrics_dir) / "walkforward_fold_summary.csv", index=False)
    all_preds_eval.to_csv(Path(config.logging.metrics_dir) / "all_test_predictions.csv", index=False)

    table = Table(title="Walk-forward Final Summary")
    table.add_column("Fold", justify="right")
    table.add_column("Rank IC", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Ann Return", justify="right")
    table.add_column("Max DD", justify="right")

    for row in fold_metrics:
        table.add_row(
            str(row.get("fold", "-")),
            f"{row.get('rank_ic_mean', np.nan):.4f}",
            f"{row.get('sharpe_ratio', np.nan):.3f}",
            f"{row.get('annualized_return', np.nan):.2%}",
            f"{row.get('max_drawdown', np.nan):.2%}",
        )

    agg = all_metrics.get("overall", {})
    table.add_row(
        "ALL",
        f"{agg.get('rank_ic_mean', np.nan):.4f}",
        f"{agg.get('sharpe_ratio', np.nan):.3f}",
        f"{agg.get('annualized_return', np.nan):.2%}",
        f"{agg.get('max_drawdown', np.nan):.2%}",
    )
    Console().print(table)
    logger.info("end_timestamp=%s runtime_sec=%.2f", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.time() - t0)


if __name__ == "__main__":
    main()
