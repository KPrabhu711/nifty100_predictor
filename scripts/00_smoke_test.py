"""
Usage: python scripts/00_smoke_test.py [--strict]

Smoke checks for core pipeline readiness:
- raw/processed data coverage
- feature NaN sanity
- dataset shapes
- single-date and batched model forward
- finite fp16 multi-task loss
- walk-forward split overlap checks
- plot artifact presence
"""

from __future__ import annotations

import argparse
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
from torch.cuda.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import NIFTY100Dataset, build_dataloaders
from src.losses.losses import MultiTaskLoss
from src.models.full_model import NIFTY100PredictionModel
from src.training.walkforward import generate_walkforward_splits
from src.utils.logging_utils import setup_logger


EPS = 1e-8


def _status(passed: bool, critical: bool) -> str:
    if passed:
        return "PASS"
    return "FAIL" if critical else "WARN"


def _add_result(results: list[dict], name: str, passed: bool, detail: str, critical: bool = True):
    results.append({"name": name, "passed": passed, "detail": detail, "critical": critical})


def _check_plot_artifacts(plots_dir: Path) -> tuple[bool, str]:
    required_exact = [
        "pretrain_loss.png",
        "walkforward_summary.png",
        "ic_over_time.png",
        "ic_distribution.png",
        "portfolio_cumulative_returns.png",
        "portfolio_drawdown.png",
        "top_bottom_spread.png",
        "feature_importance.png",
        "sector_ic.png",
        "embedding_similarity_heatmap.png",
    ]
    required_glob = [
        "fold_*_train_val_loss.png",
        "fold_*_loss_components.png",
    ]

    missing_exact = [name for name in required_exact if not (plots_dir / name).exists()]
    missing_glob = [pat for pat in required_glob if len(list(plots_dir.glob(pat))) == 0]

    passed = len(missing_exact) == 0 and len(missing_glob) == 0
    if passed:
        return True, "all expected plot artifacts found"
    missing = ", ".join(missing_exact + missing_glob)
    return False, f"missing plot artifacts: {missing}"


def main():
    parser = argparse.ArgumentParser(description="NIFTY100 pipeline smoke tests")
    parser.add_argument("--strict", action="store_true", help="fail on warnings as well")
    args = parser.parse_args()

    t0 = time.time()
    config = OmegaConf.load(PROJECT_ROOT / "config" / "config.yaml")
    logger = setup_logger(config)
    console = Console()
    results: list[dict] = []

    universe = pd.read_csv(PROJECT_ROOT / config.data.universe_file)
    tickers = universe["ticker"].dropna().astype(str).tolist()
    raw_dir = PROJECT_ROOT / config.data.raw_dir
    processed_dir = PROJECT_ROOT / config.data.processed_dir
    plots_dir = PROJECT_ROOT / config.logging.plots_dir

    raw_missing = [t for t in tickers if not (raw_dir / f"{t}.parquet").exists()]
    raw_ok = len(raw_missing) == 0
    _add_result(
        results,
        "Raw Data Coverage",
        raw_ok,
        f"missing={len(raw_missing)} / total={len(tickers)}",
        critical=True,
    )

    market_ok = (raw_dir / "NIFTY100_INDEX.parquet").exists() and (raw_dir / "INDIA_VIX.parquet").exists()
    _add_result(results, "Market Files", market_ok, "NIFTY100_INDEX + INDIA_VIX present", critical=True)

    proc_missing = [t for t in tickers if not (processed_dir / f"{t}_features.parquet").exists()]
    proc_ok = len(proc_missing) == 0
    _add_result(
        results,
        "Processed Features Coverage",
        proc_ok,
        f"missing={len(proc_missing)} / total={len(tickers)}",
        critical=True,
    )

    aux_ok = all(
        [
            (processed_dir / "regime_features.parquet").exists(),
            (processed_dir / "sector_map.parquet").exists(),
            (processed_dir / "feature_columns.json").exists(),
        ]
    )
    _add_result(results, "Processed Aux Files", aux_ok, "regime_features + sector_map + feature_columns present", critical=True)

    nan_pass = False
    nan_detail = "feature files unavailable"
    feature_cols = []
    feat_json = processed_dir / "feature_columns.json"
    if feat_json.exists():
        with feat_json.open("r", encoding="utf-8") as fp:
            feature_cols = json.load(fp)
    if feature_cols and proc_ok:
        ratios = []
        for t in tickers:
            fp = processed_dir / f"{t}_features.parquet"
            if not fp.exists():
                continue
            df = pd.read_parquet(fp, columns=feature_cols)
            ratios.append(float(df.isna().mean().mean()))

        if ratios:
            mean_nan = float(np.mean(ratios))
            max_nan = float(np.max(ratios))
            nan_pass = max_nan < 0.20
            nan_detail = f"mean_nan={mean_nan:.4f}, max_nan={max_nan:.4f}"

    _add_result(results, "NaN Pattern Sanity", nan_pass, nan_detail, critical=True)

    ds = None
    sample = None
    shape_pass = False
    shape_detail = "dataset not built"
    try:
        ds = NIFTY100Dataset(config)
        if len(ds) > 0:
            sample = ds[0]
            n, l, f = sample["features"].shape
            cond = (
                n == ds.n_stocks
                and l == int(config.features.lookback)
                and f == len(ds.feature_columns)
                and sample["targets_reg"].shape[0] == ds.n_stocks
                and sample["targets_dir"].shape[0] == ds.n_stocks
                and sample["sector_ids"].shape[0] == ds.n_stocks
            )
            shape_pass = bool(cond)
            shape_detail = f"N={n}, L={l}, F={f}, dates={len(ds)}"
        else:
            shape_detail = "dataset has zero valid dates"
    except Exception as exc:
        shape_detail = f"dataset error: {exc}"

    _add_result(results, "Dataset Shapes", shape_pass, shape_detail, critical=True)

    model = None
    single_pass = False
    single_detail = "model not run"
    loss_pass = False
    loss_detail = "loss not computed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sample is not None and ds is not None:
        try:
            model = NIFTY100PredictionModel(config=config, feature_dim=len(ds.feature_columns)).to(device)
            model.eval()
            n_classes = int(getattr(config.model, "direction_n_classes", 2))

            x = sample["features"].to(device)
            regime = sample["regime"].to(device)
            sid = sample["sector_ids"].to(device)
            rr = sample["raw_returns"].to(device)

            if device.type == "cpu":
                x = x.float()
                regime = regime.float()

            with torch.no_grad():
                with autocast(enabled=bool(config.training.fp16) and device.type == "cuda"):
                    out = model(features=x, regime=regime, graph_dict={"sector_ids": sid}, raw_returns=rr)

            n = ds.n_stocks
            cond = (
                out["ret_pred"].shape == (n,)
                and out["rank_score"].shape == (n,)
                and out["dir_logits"].shape == (n, n_classes)
                and out["embeddings"].shape == (n, int(config.model.embedding_dim))
            )
            single_pass = bool(cond)
            single_detail = (
                f"ret={tuple(out['ret_pred'].shape)}, rank={tuple(out['rank_score'].shape)}, "
                f"dir={tuple(out['dir_logits'].shape)}"
            )

            criterion = MultiTaskLoss(config).to(device)
            y_reg = sample["targets_reg"].to(device).float()
            y_reg = (y_reg - y_reg.mean()) / (y_reg.std(unbiased=False) + EPS)
            y_dir = sample["targets_dir"].to(device).long()

            with torch.no_grad():
                with autocast(enabled=bool(config.training.fp16) and device.type == "cuda"):
                    loss, comps = criterion(out, y_reg, y_dir)
            loss_val = float(loss.detach().cpu().item())
            loss_pass = bool(np.isfinite(loss_val))
            loss_detail = f"loss={loss_val:.6f}, reg={comps['loss_reg']:.6f}, rank={comps['loss_rank']:.6f}, dir={comps['loss_dir']:.6f}"
        except Exception as exc:
            single_detail = f"forward error: {exc}"
            loss_detail = f"loss error: {exc}"

    _add_result(results, "Single-Date Forward", single_pass, single_detail, critical=True)
    _add_result(results, "Finite Multi-Task Loss", loss_pass, loss_detail, critical=True)

    batch_pass = False
    batch_detail = "batch forward not run"
    if ds is not None and model is not None and len(ds) > 0:
        try:
            date_pool = [ds.calendar[i].strftime("%Y-%m-%d") for i in ds.valid_idxs[: max(8, int(config.training.batch_dates))]]
            dl = build_dataloaders(config, date_pool)
            batch = next(iter(dl))
            b = int(batch["features"].shape[0])

            with torch.no_grad():
                for i in range(b):
                    x = batch["features"][i].to(device)
                    regime = batch["regime"][i].to(device)
                    sid = batch["sector_ids"][i].to(device)
                    rr = batch["raw_returns"][i].to(device)
                    if device.type == "cpu":
                        x = x.float()
                        regime = regime.float()
                    with autocast(enabled=bool(config.training.fp16) and device.type == "cuda"):
                        _ = model(features=x, regime=regime, graph_dict={"sector_ids": sid}, raw_returns=rr)

            batch_pass = True
            batch_detail = f"forwarded {b} dates without error"
        except Exception as exc:
            batch_detail = f"batch forward error: {exc}"

    _add_result(results, "Batch Forward", batch_pass, batch_detail, critical=True)

    wf_pass = False
    wf_detail = "walk-forward splits not checked"
    wf_cross_pass = True
    wf_cross_detail = "cross-fold overlap not checked"
    if ds is not None:
        try:
            all_dates = [ds.calendar[i].strftime("%Y-%m-%d") for i in ds.valid_idxs]
            splits = generate_walkforward_splits(all_dates, config)
            ok = True
            for sp in splits:
                tr, va, te = set(sp["train_dates"]), set(sp["val_dates"]), set(sp["test_dates"])
                if tr & va or tr & te or va & te:
                    ok = False
                    break

            wf_pass = ok and len(splits) > 0
            wf_detail = f"folds={len(splits)}, intra-fold overlap={'none' if ok else 'detected'}"

            overlap_pairs = 0
            for i in range(len(splits)):
                si = set(splits[i]["test_dates"])
                for j in range(i + 1, len(splits)):
                    sj = set(splits[j]["test_dates"])
                    if si & sj:
                        overlap_pairs += 1

            wf_cross_pass = overlap_pairs == 0
            wf_cross_detail = f"cross-fold test overlap pairs={overlap_pairs}"
        except Exception as exc:
            wf_detail = f"walk-forward error: {exc}"
            wf_cross_pass = False
            wf_cross_detail = f"walk-forward error: {exc}"

    _add_result(results, "Walk-Forward Intra-Fold", wf_pass, wf_detail, critical=True)
    _add_result(results, "Walk-Forward Cross-Fold", wf_cross_pass, wf_cross_detail, critical=False)

    plot_pass, plot_detail = _check_plot_artifacts(plots_dir)
    _add_result(results, "Plot Artifacts", plot_pass, plot_detail, critical=False)

    table = Table(title="NIFTY100 Smoke Test")
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Critical", justify="center")
    table.add_column("Details")

    for row in results:
        table.add_row(
            row["name"],
            _status(row["passed"], row["critical"]),
            "yes" if row["critical"] else "no",
            row["detail"],
        )

    console.print(table)

    critical_fail = any((not r["passed"]) and r["critical"] for r in results)
    any_fail = any(not r["passed"] for r in results)
    should_fail = critical_fail or (args.strict and any_fail)

    logger.info("smoke_test_complete critical_fail=%s any_fail=%s strict=%s", critical_fail, any_fail, args.strict)
    logger.info("end_timestamp=%s runtime_sec=%.2f", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.time() - t0)

    raise SystemExit(1 if should_fail else 0)


if __name__ == "__main__":
    main()
