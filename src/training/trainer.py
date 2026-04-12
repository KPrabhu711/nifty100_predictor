from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast

from src.evaluation.metrics import apply_alpha_score, compute_all_metrics, select_alpha_configuration
from src.losses.losses import MultiTaskLoss
from src.utils.plotting import plot_loss_components, plot_train_val_loss


EPS = 1e-8


class Trainer:
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.training.supervised_lr),
            weight_decay=float(config.training.weight_decay),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(config.training.supervised_epochs),
            eta_min=float(config.training.supervised_lr) * 0.1,
        )
        self.scaler = GradScaler(enabled=bool(config.training.fp16) and self.device.type == "cuda")
        self.criterion = MultiTaskLoss(config)

        self.early_patience = int(config.training.early_stopping_patience)
        self.early_metric = str(config.training.early_stopping_metric)
        self.log_every = int(config.logging.log_every_n_steps)
        self.train_date_weights: dict[str, float] = {}

    def _configure_direction_class_weights(self, dataset) -> None:
        n_classes = int(getattr(self.config.model, "direction_n_classes", 2))
        counts = np.zeros(n_classes, dtype=np.float64)

        for t_idx in dataset.valid_idxs:
            mask = dataset.target_mask[:, t_idx] > 0.5
            if not np.any(mask):
                continue
            labels = dataset.targets_dir[:, t_idx][mask]
            labels = labels[np.isfinite(labels)].astype(np.int64)
            labels = labels[(labels >= 0) & (labels < n_classes)]
            if labels.size == 0:
                continue
            binc = np.bincount(labels, minlength=n_classes)
            counts += binc

        total = float(counts.sum())
        if total <= 0:
            self.logger.warning("direction_class_weights_skipped reason=no_valid_labels")
            return

        if self.criterion.auto_class_weights:
            beta = 0.999
            effective_num = 1.0 - np.power(beta, counts)
            effective_num = np.where(effective_num <= 0.0, 1e-12, effective_num)
            weights = (1.0 - beta) / effective_num
            weights = weights / (weights.mean() + 1e-12)
            weights = np.clip(weights, self.criterion.class_weight_clip_min, self.criterion.class_weight_clip_max)
            self.criterion.set_focal_alpha(weights.tolist())
            self.logger.info(
                "direction_class_weights_applied counts=%s weights=%s",
                np.round(counts, 2).tolist(),
                np.round(weights, 4).tolist(),
            )
        else:
            self.logger.info("direction_class_weights_static counts=%s", np.round(counts, 2).tolist())

    def _zscore_cross_section(self, target_reg: torch.Tensor, target_mask: torch.Tensor | None = None) -> torch.Tensor:
        if target_mask is None:
            mean = target_reg.mean()
            std = target_reg.std(unbiased=False)
            return (target_reg - mean) / (std + EPS)

        m = target_mask > 0.5
        if torch.sum(m) < 2:
            return torch.zeros_like(target_reg)

        z = torch.zeros_like(target_reg)
        mean = target_reg[m].mean()
        std = target_reg[m].std(unbiased=False)
        z[m] = (target_reg[m] - mean) / (std + EPS)
        return z

    def _build_train_date_weights(self, dataset) -> None:
        self.train_date_weights = {}
        if not bool(getattr(self.config.training, "recency_weighting", False)):
            return

        half_life = float(getattr(self.config.training, "recency_half_life_days", 252.0))
        dates = [dataset.calendar[i].strftime("%Y-%m-%d") for i in dataset.valid_idxs]
        if not dates:
            return

        n = len(dates)
        steps_to_end = np.arange(n - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-np.log(2.0) * steps_to_end / max(half_life, 1.0))
        weights = weights / (weights.mean() + EPS)
        self.train_date_weights = {d: float(w) for d, w in zip(dates, weights)}
        self.logger.info(
            "recency_weights_applied half_life=%.1f min=%.4f max=%.4f",
            half_life,
            float(weights.min()),
            float(weights.max()),
        )

    def _resolve_early_metric_key(self) -> str:
        mapping = {
            "rank_ic": "rank_ic_mean",
            "score_rank_ic": "score_rank_ic_mean",
            "alpha_rank_ic": "score_rank_ic_mean",
            "sharpe": "sharpe_ratio",
            "alpha_sharpe": "sharpe_ratio",
            "spread": "spread_mean",
            "top_k_precision": "top_k_precision",
        }
        return mapping.get(self.early_metric, self.early_metric)

    def train_epoch(self, dataloader):
        self.model.train()
        loss_hist = []
        reg_hist, rank_hist, dir_hist = [], [], []

        for step, batch in enumerate(dataloader):
            self.optimizer.zero_grad(set_to_none=True)

            features = batch["features"]
            targets_reg = batch["targets_reg"]
            targets_dir = batch["targets_dir"]
            target_mask = batch["target_mask"]
            regime = batch["regime"]
            sector_ids = batch["sector_ids"]
            raw_returns = batch["raw_returns"]
            dates = batch["date"]

            bsz = features.size(0)
            accum_loss = torch.tensor(0.0, device=self.device)
            comp_reg, comp_rank, comp_dir = [], [], []

            for i in range(bsz):
                x = features[i].to(self.device, non_blocking=True)
                y_reg_raw = targets_reg[i].to(self.device, non_blocking=True)
                y_dir = targets_dir[i].to(self.device, non_blocking=True)
                y_mask = target_mask[i].to(self.device, non_blocking=True)
                y_reg = self._zscore_cross_section(y_reg_raw, y_mask)
                rg = regime[i].to(self.device, non_blocking=True)
                sid = sector_ids[i].to(self.device, non_blocking=True)
                rr = raw_returns[i].to(self.device, non_blocking=True)

                with autocast(enabled=bool(self.config.training.fp16) and self.device.type == "cuda"):
                    out = self.model(
                        features=x,
                        regime=rg,
                        graph_dict={"sector_ids": sid},
                        raw_returns=rr,
                    )
                    loss, comps = self.criterion(out, y_reg, y_dir, y_mask)

                date_weight = float(self.train_date_weights.get(str(dates[i]), 1.0))
                loss = loss * date_weight

                accum_loss = accum_loss + loss
                comp_reg.append(comps["loss_reg"])
                comp_rank.append(comps["loss_rank"])
                comp_dir.append(comps["loss_dir"])

            accum_loss = accum_loss / max(bsz, 1)

            self.scaler.scale(accum_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            step_loss = float(accum_loss.detach().cpu().item())
            loss_hist.append(step_loss)
            reg_hist.append(float(np.mean(comp_reg)))
            rank_hist.append(float(np.mean(comp_rank)))
            dir_hist.append(float(np.mean(comp_dir)))

            if step % self.log_every == 0:
                self.logger.log(
                    {
                        "train_step_loss": step_loss,
                        "train_loss_reg": reg_hist[-1],
                        "train_loss_rank": rank_hist[-1],
                        "train_loss_dir": dir_hist[-1],
                    },
                    step=step,
                )

        return {
            "loss_total": float(np.mean(loss_hist)) if loss_hist else np.nan,
            "loss_reg": float(np.mean(reg_hist)) if reg_hist else np.nan,
            "loss_rank": float(np.mean(rank_hist)) if rank_hist else np.nan,
            "loss_dir": float(np.mean(dir_hist)) if dir_hist else np.nan,
            "components": {
                "reg": reg_hist,
                "rank": rank_hist,
                "dir": dir_hist,
            },
        }

    @torch.no_grad()
    def val_epoch(self, dataloader, alpha_selection: dict | None = None):
        self.model.eval()

        rows = []
        losses = []
        ticker_list = dataloader.dataset.tickers

        for batch in dataloader:
            features = batch["features"]
            targets_reg = batch["targets_reg"]
            targets_dir = batch["targets_dir"]
            target_mask = batch["target_mask"]
            regime = batch["regime"]
            sector_ids = batch["sector_ids"]
            raw_returns = batch["raw_returns"]
            dates = batch["date"]

            bsz = features.size(0)
            for i in range(bsz):
                x = features[i].to(self.device, non_blocking=True)
                y_reg_raw = targets_reg[i].to(self.device, non_blocking=True)
                y_dir = targets_dir[i].to(self.device, non_blocking=True)
                y_mask = target_mask[i].to(self.device, non_blocking=True)
                y_reg = self._zscore_cross_section(y_reg_raw, y_mask)
                rg = regime[i].to(self.device, non_blocking=True)
                sid = sector_ids[i].to(self.device, non_blocking=True)
                rr = raw_returns[i].to(self.device, non_blocking=True)

                with autocast(enabled=bool(self.config.training.fp16) and self.device.type == "cuda"):
                    out = self.model(
                        features=x,
                        regime=rg,
                        graph_dict={"sector_ids": sid},
                        raw_returns=rr,
                    )
                    loss, _ = self.criterion(out, y_reg, y_dir, y_mask)

                losses.append(float(loss.detach().cpu().item()))

                pred_ret = out["ret_pred"].detach().float().cpu().numpy()
                pred_rank = out["rank_score"].detach().float().cpu().numpy()
                dir_probs = torch.softmax(out["dir_logits"].detach().float().cpu(), dim=1).numpy()
                pred_dir = dir_probs.argmax(axis=1)
                actual_ret = y_reg_raw.detach().float().cpu().numpy()
                actual_dir = y_dir.detach().long().cpu().numpy()
                valid_mask = y_mask.detach().float().cpu().numpy() > 0.5
                date_str = dates[i]

                for j, ticker in enumerate(ticker_list):
                    if not valid_mask[j]:
                        continue
                    rows.append(
                        {
                            "date": date_str,
                            "ticker": ticker,
                            "pred_return": float(pred_ret[j]),
                            "pred_rank": float(pred_rank[j]),
                            "pred_dir": int(pred_dir[j]),
                            "pred_dir_prob_down": float(dir_probs[j, 0]),
                            "pred_dir_prob_flat": float(dir_probs[j, 1]) if dir_probs.shape[1] > 2 else 0.0,
                            "pred_dir_prob_up": float(dir_probs[j, -1]),
                            "pred_dir_score": float(dir_probs[j, -1] - dir_probs[j, 0]),
                            "actual_return": float(actual_ret[j]),
                            "actual_dir": int(actual_dir[j]),
                        }
                    )

        pred_df = pd.DataFrame(rows)
        alpha_cfg = alpha_selection
        eval_df = pred_df
        if not pred_df.empty:
            if alpha_cfg is None:
                alpha_cfg = select_alpha_configuration(pred_df, self.config)
                eval_df = alpha_cfg["predictions_df"]
            else:
                weights = alpha_cfg.get("weights", (1.0, 0.0, 0.0))
                eval_df = apply_alpha_score(
                    pred_df,
                    weights=weights,
                    out_col=alpha_cfg.get("score_col", "alpha_score"),
                    use_zscore=bool(getattr(self.config.evaluation, "alpha_use_zscore", True)),
                    direction_component=str(getattr(self.config.evaluation, "alpha_direction_component", "up_minus_down")),
                )

        eval_config = self.config
        if not pred_df.empty and alpha_cfg is not None:
            eval_config = self.config.copy()
            eval_config.evaluation.portfolio_score_col = alpha_cfg.get("score_col", str(self.config.evaluation.portfolio_score_col))

        metrics = compute_all_metrics(eval_df, eval_config) if not eval_df.empty else {}
        overall = metrics.get("overall", {})
        out = {
            "val_loss": float(np.mean(losses)) if losses else np.nan,
            "ic_mean": overall.get("ic_mean", np.nan),
            "rank_ic_mean": overall.get("rank_ic_mean", np.nan),
            "score_rank_ic_mean": overall.get("score_rank_ic_mean", np.nan),
            "directional_accuracy": overall.get("directional_accuracy", np.nan),
            "top_k_precision": overall.get("top_k_precision", np.nan),
            "predictions_df": eval_df,
            "metrics": metrics,
            "alpha_selection": alpha_cfg,
        }
        return out

    def train_fold(self, train_dl, val_dl, fold_id):
        ckpt_dir = Path(self.config.logging.checkpoint_dir)
        plot_dir = Path(self.config.logging.plots_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        best_metric = -np.inf
        patience = 0
        best_val = None

        train_losses = []
        val_losses = []
        reg_comp, rank_comp, dir_comp = [], [], []

        self._configure_direction_class_weights(train_dl.dataset)
        self._build_train_date_weights(train_dl.dataset)

        for epoch in range(int(self.config.training.supervised_epochs)):
            train_stats = self.train_epoch(train_dl)
            val_stats = self.val_epoch(val_dl)
            self.scheduler.step()

            train_losses.append(train_stats["loss_total"])
            val_losses.append(val_stats["val_loss"])

            reg_comp.extend(train_stats["components"]["reg"])
            rank_comp.extend(train_stats["components"]["rank"])
            dir_comp.extend(train_stats["components"]["dir"])

            log_payload = {
                "epoch": epoch,
                "train_loss": train_stats["loss_total"],
                "val_loss": val_stats["val_loss"],
                "val_ic": val_stats["ic_mean"],
                "val_rank_ic": val_stats["rank_ic_mean"],
                "val_score_rank_ic": val_stats["score_rank_ic_mean"],
                "val_directional_acc": val_stats["directional_accuracy"],
                "val_top_k_precision": val_stats["top_k_precision"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.logger.log(log_payload, step=epoch)

            metric_key = self._resolve_early_metric_key()
            metric = float(val_stats.get(metric_key, np.nan))
            if np.isnan(metric):
                metric = -np.inf

            if best_val is None or metric > best_metric:
                best_metric = metric
                patience = 0
                best_val = val_stats
                ckpt_path = ckpt_dir / f"fold_{fold_id}_best.pt"

                overall = val_stats.get("metrics", {}).get("overall", {})
                overall_clean = {}
                if isinstance(overall, dict):
                    for k, v in overall.items():
                        try:
                            overall_clean[k] = float(v)
                        except Exception:
                            continue

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_metric": best_metric,
                        "val_overall": overall_clean,
                    },
                    ckpt_path,
                )
            else:
                patience += 1

            if patience >= self.early_patience:
                self.logger.info("early_stopping fold=%s epoch=%s best_rank_ic=%.5f", fold_id, epoch, best_metric)
                break

        plot_train_val_loss(train_losses, val_losses, fold_id, plot_dir / f"fold_{fold_id}_train_val_loss.png")
        plot_loss_components(reg_comp, rank_comp, dir_comp, fold_id, plot_dir / f"fold_{fold_id}_loss_components.png")

        return best_val

    @torch.no_grad()
    def infer(self, dataloader, alpha_selection: dict | None = None):
        val_out = self.val_epoch(dataloader, alpha_selection=alpha_selection)
        return val_out["predictions_df"], val_out.get("metrics", {})
