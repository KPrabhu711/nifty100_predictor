from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def huber_loss(pred, target, delta: float = 0.01, mask: torch.Tensor | None = None):
    """Standard Huber loss."""
    err = pred - target
    abs_err = err.abs()
    quad = 0.5 * err.pow(2)
    lin = delta * (abs_err - 0.5 * delta)
    loss = torch.where(abs_err <= delta, quad, lin)
    if mask is not None:
        m = mask > 0.5
        if not torch.any(m):
            return pred.new_tensor(0.0)
        return loss[m].mean()
    return loss.mean()


def pairwise_ranking_loss(
    scores,
    targets,
    margin: float = 0.05,
    pair_epsilon: float = 1e-4,
    mask: torch.Tensor | None = None,
    loss_type: str = "logistic",
):
    """
    Pairwise ranking loss over ordered target pairs.

    loss_type:
      - "hinge": relu(margin - (s_i - s_j))
      - "logistic": softplus(margin - (s_i - s_j))
    """
    if mask is not None:
        m = mask > 0.5
        if torch.sum(m) < 2:
            return scores.new_tensor(0.0)
        scores = scores[m]
        targets = targets[m]

    tdiff = targets.unsqueeze(1) - targets.unsqueeze(0)
    sdiff = scores.unsqueeze(1) - scores.unsqueeze(0)

    valid = tdiff > pair_epsilon
    if not valid.any():
        return scores.new_tensor(0.0)

    margins = margin - sdiff[valid]
    if str(loss_type).lower() == "hinge":
        base = F.relu(margins)
    else:
        base = F.softplus(margins)

    weights = torch.clamp(torch.abs(tdiff[valid]), min=0.0, max=3.0)
    weights = weights / (weights.mean().detach() + EPS)
    loss = (base * weights).mean()
    return loss


def focal_loss(
    logits,
    labels,
    gamma: float = 2.0,
    mask: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
):
    """
    Multi-class focal loss.
    logits: [N, C], labels: [N] in {0, ..., C-1}
    """
    if mask is not None:
        m = mask > 0.5
        if torch.sum(m) == 0:
            return logits.new_tensor(0.0)
        logits = logits[m]
        labels = labels[m]

    labels = labels.long()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    idx = labels.unsqueeze(1)
    log_pt = log_probs.gather(1, idx).squeeze(1)
    pt = probs.gather(1, idx).squeeze(1).clamp(min=EPS, max=1.0)

    ce = -log_pt
    fl = (1.0 - pt).pow(gamma) * ce

    if alpha is not None:
        alpha_vec = alpha.to(logits.device)
        alpha_t = alpha_vec[labels]
        fl = fl * alpha_t

    return fl.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task objective.
    """

    def __init__(self, config):
        super().__init__()
        self.lambda_reg = float(config.loss.lambda_reg)
        self.lambda_rank = float(config.loss.lambda_rank)
        self.lambda_dir = float(config.loss.lambda_dir)
        self.huber_delta = float(config.loss.huber_delta)
        self.ranking_margin = float(getattr(config.loss, "ranking_margin", 0.05))
        self.ranking_pair_epsilon = float(getattr(config.loss, "ranking_pair_epsilon", 1e-4))
        self.ranking_loss_type = str(getattr(config.loss, "ranking_loss_type", "logistic"))
        self.focal_gamma = float(config.loss.focal_gamma)
        self.auto_class_weights = bool(getattr(config.loss, "auto_class_weights", False))
        self.class_weight_clip_min = float(getattr(config.loss, "class_weight_clip_min", 0.5))
        self.class_weight_clip_max = float(getattr(config.loss, "class_weight_clip_max", 3.0))

        focal_alpha = getattr(config.loss, "focal_alpha", None)
        if focal_alpha is None:
            self.focal_alpha = None
        else:
            self.focal_alpha = torch.tensor(list(focal_alpha), dtype=torch.float32)

    def set_focal_alpha(self, alpha_values) -> None:
        if alpha_values is None:
            self.focal_alpha = None
            return
        alpha_t = torch.tensor(list(alpha_values), dtype=torch.float32)
        alpha_t = torch.clamp(alpha_t, min=self.class_weight_clip_min, max=self.class_weight_clip_max)
        self.focal_alpha = alpha_t

    def forward(
        self,
        outputs: dict,
        target_reg: torch.Tensor,
        target_dir: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ):
        reg = huber_loss(outputs["ret_pred"], target_reg, delta=self.huber_delta, mask=target_mask)
        rank = pairwise_ranking_loss(
            outputs["rank_score"],
            target_reg,
            margin=self.ranking_margin,
            pair_epsilon=self.ranking_pair_epsilon,
            mask=target_mask,
            loss_type=self.ranking_loss_type,
        )
        direction = focal_loss(
            outputs["dir_logits"],
            target_dir,
            gamma=self.focal_gamma,
            mask=target_mask,
            alpha=self.focal_alpha,
        )

        total = self.lambda_reg * reg + self.lambda_rank * rank + self.lambda_dir * direction
        metrics = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_reg": float(reg.detach().cpu().item()),
            "loss_rank": float(rank.detach().cpu().item()),
            "loss_dir": float(direction.detach().cpu().item()),
        }
        return total, metrics
