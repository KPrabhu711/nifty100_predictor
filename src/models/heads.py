from __future__ import annotations

import torch.nn as nn


class ReturnRegressionHead(nn.Module):
    """
    MLP: D + regime_dim -> 256 -> 128 -> 1.
    """

    def __init__(self, in_dim: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RankingHead(nn.Module):
    """
    MLP: D + regime_dim -> 128 -> 1.
    """

    def __init__(self, in_dim: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DirectionHead(nn.Module):
    """
    MLP: D + regime_dim -> 128 -> C.
    """

    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)
