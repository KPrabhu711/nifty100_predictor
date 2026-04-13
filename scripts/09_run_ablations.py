"""
Usage: python scripts/09_run_ablations.py

Runs a compact ablation suite using the same walk-forward protocol as the main model.
Ablations:
  - no_pretrain
  - no_ranking_loss
  - no_regime
  - temporal_only
"""

from __future__ import annotations

import copy
import json
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
from src.models.full_model import NIFTY100PredictionModel
from src.training.experiment_utils import (
    dedupe_overlap_predictions,
    load_full_model_reference,
    save_comparison_plot,
    set_seed,
)
from src.training.trainer import Trainer
from src.training.walkforward import generate_walkforward_splits
from src.utils.logging_utils import setup_logger


def _set_nested(cfg, dotted_key: str, value):
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], value)


def _variant_specs():
    return [
        ("no_pretrain", {"training.use_pretrained_backbone": False}),
        ("no_ranking_loss", {"loss.lambda_rank": 0.0}),
        ("no_regime", {"model.use_regime": False}),
        ("temporal_only", {"model.use_graph": False}),
    ]


def main() -> None:
    t0 = time.time()
    set_seed(42)
    base_config = OmegaConf.load(PROJECT_ROOT / "config" / "config.yaml")
    logger = setup_logger(base_config)

    base_dataset = NIFTY100Dataset(base_config)
    all_dates = [
        base_dataset.calendar[i].strftime("%Y-%m-%d") for i in base_dataset.valid_idxs
    ]
    splits = generate_walkforward_splits(all_dates, base_config)
    if not splits:
        raise RuntimeError("No walk-forward splits generated")

    pretrain_ckpt = (
        PROJECT_ROOT / base_config.logging.checkpoint_dir / "pretrain_best.pt"
    )
    ablation_metrics_dir = PROJECT_ROOT / "results" / "metrics" / "ablations"
    ablation_metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for variant_name, overrides in _variant_specs():
        cfg = copy.deepcopy(base_config)
        for k, v in overrides.items():
            _set_nested(cfg, k, v)

        cfg.logging.checkpoint_dir = str(
            PROJECT_ROOT / "results" / "checkpoints" / "ablations" / variant_name
        )
        cfg.logging.plots_dir = str(
            PROJECT_ROOT / "results" / "plots" / "ablations" / variant_name
        )

        logger.info("ablation_start variant=%s overrides=%s", variant_name, overrides)
        all_preds = []
        fold_rows = []

        for split in splits:
            fold = split["fold"]
            cfg.runtime_split_dates = {
                "train": split["train_dates"],
                "val": split["val_dates"],
                "test": split["test_dates"],
            }

            train_dl = build_dataloaders(cfg, "train")
            val_dl = build_dataloaders(cfg, "val")
            test_dl = build_dataloaders(cfg, "test")
            if (
                len(train_dl.dataset) == 0
                or len(val_dl.dataset) == 0
                or len(test_dl.dataset) == 0
            ):
                logger.warning(
                    "skip_ablation_fold_empty variant=%s fold=%d", variant_name, fold
                )
                continue

            model = NIFTY100PredictionModel(
                cfg, feature_dim=len(train_dl.dataset.feature_columns)
            )
            if (
                bool(getattr(cfg.training, "use_pretrained_backbone", True))
                and pretrain_ckpt.exists()
            ):
                payload = torch.load(
                    pretrain_ckpt, map_location="cpu", weights_only=False
                )
                model.temporal_encoder.load_state_dict(
                    payload["backbone_state_dict"], strict=False
                )

            trainer = Trainer(model=model, config=cfg, logger=logger)
            trainer.train_fold(train_dl=train_dl, val_dl=val_dl, fold_id=fold)

            ckpt_path = Path(cfg.logging.checkpoint_dir) / f"fold_{fold}_best.pt"
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            trainer.model.load_state_dict(state["model_state_dict"], strict=True)

            _, val_metrics = trainer.infer(val_dl)
            alpha_selection = (
                val_metrics.get("alpha_selection", {})
                if isinstance(val_metrics, dict)
                else {}
            )
            pred_df, test_metrics = trainer.infer(
                test_dl, alpha_selection=alpha_selection
            )
            pred_df["fold"] = fold
            pred_df.to_csv(
                ablation_metrics_dir / f"{variant_name}_fold_{fold}_predictions.csv",
                index=False,
            )

            overall = test_metrics.get("overall", {})
            overall["fold"] = fold
            overall["variant"] = variant_name
            fold_rows.append(overall)
            all_preds.append(pred_df)

            with (
                ablation_metrics_dir / f"{variant_name}_fold_{fold}_metrics.json"
            ).open("w", encoding="utf-8") as fp:
                json.dump(
                    {"overall": overall, "alpha_selection": alpha_selection},
                    fp,
                    indent=2,
                )

        if not all_preds:
            continue

        variant_preds = pd.concat(all_preds, ignore_index=True)
        variant_preds = dedupe_overlap_predictions(
            variant_preds, str(base_config.evaluation.aggregate_overlap_policy)
        )
        variant_preds.to_csv(
            ablation_metrics_dir / f"{variant_name}_all_predictions.csv", index=False
        )

        from src.evaluation.metrics import compute_all_metrics

        overall = compute_all_metrics(variant_preds, cfg)["overall"]
        overall["variant"] = variant_name
        summary_rows.append(overall)
        pd.DataFrame(fold_rows).to_csv(
            ablation_metrics_dir / f"{variant_name}_fold_summary.csv", index=False
        )
        logger.info(
            "ablation_complete variant=%s sharpe=%.4f ann_return=%.4f rank_ic=%.4f",
            variant_name,
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
        PROJECT_ROOT / "results" / "metrics" / "ablations_summary.csv", index=False
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
        PROJECT_ROOT / "results" / "plots" / "ablations_comparison.png",
        title="Ablation Comparison",
    )

    table = Table(title="Ablation Summary")
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
        "ablations_saved path=%s",
        str(PROJECT_ROOT / "results" / "metrics" / "ablations_summary.csv"),
    )
    logger.info(
        "end_timestamp=%s runtime_sec=%.2f",
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        time.time() - t0,
    )


if __name__ == "__main__":
    main()
