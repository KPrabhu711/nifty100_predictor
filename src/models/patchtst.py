from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class PatchEmbedding(nn.Module):
    """
    Converts [N, L, F] stock feature sequence into patch tokens.
    """

    def __init__(self, seq_len: int, patch_size: int, stride: int, in_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.in_dim = in_dim
        self.embed_dim = embed_dim

        self.num_patches = ((seq_len - patch_size) // stride) + 1
        self.proj = nn.Linear(patch_size * in_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        patches = patches.contiguous().view(x.size(0), self.num_patches, self.patch_size * self.in_dim)
        return patches

    def project_patches(self, patches: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(patches)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        return self.dropout(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(x)
        return self.project_patches(patches)


class PatchTSTEncoder(nn.Module):
    """
    Transformer encoder on patch tokens with optional gradient checkpointing.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(x)


class AttentionPooling(nn.Module):
    """
    Computes weighted sum of patch tokens using a learned query.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(tokens, self.query) / math.sqrt(tokens.size(-1))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled


class MultiScalePatchTST(nn.Module):
    """
    Three PatchTST branches with patch scales [8, 16, 32].
    """

    def __init__(
        self,
        seq_len: int,
        in_dim: int,
        embed_dim: int,
        patch_sizes: list[int],
        patch_strides: list[int],
        n_layers: int,
        n_heads: int,
        dropout: float,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if len(patch_sizes) != len(patch_strides):
            raise ValueError("patch_sizes and patch_strides must have same length")

        self.patch_embeddings = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.poolers = nn.ModuleList()

        for p, s in zip(patch_sizes, patch_strides):
            self.patch_embeddings.append(
                PatchEmbedding(
                    seq_len=seq_len,
                    patch_size=p,
                    stride=s,
                    in_dim=in_dim,
                    embed_dim=embed_dim,
                    dropout=dropout,
                )
            )
            self.encoders.append(
                PatchTSTEncoder(
                    embed_dim=embed_dim,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    dropout=dropout,
                    gradient_checkpointing=gradient_checkpointing,
                )
            )
            self.poolers.append(AttentionPooling(embed_dim))

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 6, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def encode_branch(self, x: torch.Tensor, branch_idx: int, tokens_override: torch.Tensor | None = None):
        pe = self.patch_embeddings[branch_idx]
        enc = self.encoders[branch_idx]

        if tokens_override is None:
            tokens = pe(x)
            patches = pe.patchify(x)
        else:
            tokens = tokens_override
            patches = pe.patchify(x)

        encoded = enc(tokens)
        pooled = self.poolers[branch_idx](encoded)
        last = encoded[:, -1, :]
        return encoded, pooled, last, patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_vecs = []
        last_vecs = []
        for i in range(len(self.patch_embeddings)):
            _, pooled, last, _ = self.encode_branch(x, i)
            pooled_vecs.append(pooled)
            last_vecs.append(last)

        fused = torch.cat(pooled_vecs + last_vecs, dim=-1)
        return self.fusion(fused)
