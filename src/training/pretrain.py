from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from src.models.patchtst import MultiScalePatchTST
from src.utils.plotting import plot_pretrain_loss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PretrainWindowDataset(Dataset):
    """
    One sample = one stock's lookback window [L, F].
    """

    def __init__(self, config):
        self.config = config
        lookback = int(config.features.lookback)
        processed_dir = Path(config.data.processed_dir)
        universe = pd.read_csv(config.data.universe_file)
        tickers = universe["ticker"].astype(str).tolist()

        feat_json = processed_dir / "feature_columns.json"
        if not feat_json.exists():
            raise FileNotFoundError("Missing data/processed/feature_columns.json. Run feature build first.")
        with feat_json.open("r", encoding="utf-8") as fp:
            self.feature_columns = json.load(fp)

        self.arrays = []
        self.index_map = []
        for ticker in tickers:
            path = processed_dir / f"{ticker}_features.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df = df.sort_values("Date")
            arr = df[self.feature_columns].to_numpy(dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if len(arr) < lookback:
                continue

            t_idx = len(self.arrays)
            self.arrays.append(arr)
            for end in range(lookback - 1, len(arr)):
                self.index_map.append((t_idx, end))

        self.lookback = lookback

    @property
    def feature_dim(self) -> int:
        return len(self.feature_columns)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        arr_idx, end = self.index_map[idx]
        arr = self.arrays[arr_idx]
        x = arr[end - self.lookback + 1 : end + 1]
        return torch.tensor(x, dtype=torch.float16)


class MaskedPatchReconstruction(nn.Module):
    """
    Multi-scale masked patch reconstruction pretraining objective.
    """

    def __init__(self, backbone: MultiScalePatchTST, in_dim: int, mask_ratio: float = 0.4):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio

        self.mask_tokens = nn.ParameterList()
        self.reconstruction_heads = nn.ModuleList()
        for pe in self.backbone.patch_embeddings:
            self.mask_tokens.append(nn.Parameter(torch.zeros(1, 1, pe.embed_dim)))
            self.reconstruction_heads.append(nn.Linear(pe.embed_dim, pe.patch_size * in_dim))

        for tok in self.mask_tokens:
            nn.init.normal_(tok, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        losses = []
        x = x.float()

        for i, pe in enumerate(self.backbone.patch_embeddings):
            enc = self.backbone.encoders[i]
            patches = pe.patchify(x)
            tokens = pe.project_patches(patches)
            b, n, _ = tokens.shape

            mask = torch.rand((b, n), device=x.device) < self.mask_ratio
            if not mask.any():
                mask[:, 0] = True

            masked_tokens = tokens.clone()
            mask_tok = self.mask_tokens[i].expand(b, n, -1)
            masked_tokens[mask] = mask_tok[mask]

            encoded = enc(masked_tokens)
            pred = self.reconstruction_heads[i](encoded[mask])
            true = patches[mask]
            losses.append(torch.mean((pred - true) ** 2))

        return torch.stack(losses).mean()


def run_pretraining(config, logger=None):
    """
    Run masked patch reconstruction pretraining and save best backbone.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PretrainWindowDataset(config)
    if logger:
        logger.info(
            "pretrain_dataset windows=%d lookback=%d feature_dim=%d",
            len(ds),
            int(config.features.lookback),
            ds.feature_dim,
        )
    dl = DataLoader(
        ds,
        batch_size=int(config.training.pretrain_batch_size),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    backbone = MultiScalePatchTST(
        seq_len=int(config.features.lookback),
        in_dim=ds.feature_dim,
        embed_dim=int(config.model.embedding_dim),
        patch_sizes=list(config.model.patch_sizes),
        patch_strides=list(config.model.patch_strides),
        n_layers=int(config.model.n_transformer_layers),
        n_heads=int(config.model.n_attention_heads),
        dropout=float(config.model.dropout),
        gradient_checkpointing=bool(config.training.gradient_checkpointing),
    )
    model = MaskedPatchReconstruction(
        backbone=backbone,
        in_dim=ds.feature_dim,
        mask_ratio=float(config.training.pretrain_mask_ratio),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.training.pretrain_lr), weight_decay=float(config.training.weight_decay))
    scaler = GradScaler(enabled=bool(config.training.fp16) and device.type == "cuda")

    epochs = int(config.training.pretrain_epochs)
    log_every = int(config.logging.log_every_n_steps)
    ckpt_dir = Path(config.logging.checkpoint_dir)
    plots_dir = Path(config.logging.plots_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    history = []
    global_step = 0

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for batch in dl:
            x = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=bool(config.training.fp16) and device.type == "cuda"):
                loss = model(x)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_losses.append(float(loss.detach().cpu().item()))

            if logger and global_step % log_every == 0:
                logger.log({"pretrain_step_loss": batch_losses[-1]}, step=global_step)
            global_step += 1

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
        history.append(epoch_loss)
        if logger:
            logger.log({"pretrain_epoch": epoch, "pretrain_loss": epoch_loss}, step=epoch)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "loss": best_loss,
                    "backbone_state_dict": model.backbone.state_dict(),
                    "feature_dim": ds.feature_dim,
                    "feature_columns": ds.feature_columns,
                },
                ckpt_dir / "pretrain_best.pt",
            )

    plot_pretrain_loss(history, plots_dir / "pretrain_loss.png")
    if logger:
        logger.info("pretrain_complete best_loss=%.6f ckpt=%s", best_loss, str(ckpt_dir / "pretrain_best.pt"))

    return {
        "best_loss": best_loss,
        "loss_history": history,
        "checkpoint_path": str(ckpt_dir / "pretrain_best.pt"),
    }
