from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class _RelationalGATLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads")

        head_dim = embed_dim // n_heads
        self.sector_gat = GATConv(
            embed_dim,
            head_dim,
            heads=n_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=False,
        )
        self.corr_gat = GATConv(
            embed_dim,
            head_dim,
            heads=n_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=False,
            edge_dim=1,
        )
        self.emb_gat = GATConv(
            embed_dim,
            head_dim,
            heads=n_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=False,
            edge_dim=1,
        )

        self.norm_sector = nn.LayerNorm(embed_dim)
        self.norm_corr = nn.LayerNorm(embed_dim)
        self.norm_emb = nn.LayerNorm(embed_dim)

        self.fusion_logits = nn.Parameter(torch.zeros(3))
        self.dropout = nn.Dropout(dropout)

    def _relation_forward(self, conv: GATConv, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return x
        if edge_weight is None:
            return conv(x, edge_index)
        edge_attr = edge_weight.view(-1, 1)
        return conv(x, edge_index, edge_attr=edge_attr)

    def forward(self, h: torch.Tensor, graph_dict: dict) -> torch.Tensor:
        sec_idx, _ = graph_dict.get("sector", (None, None))
        corr_idx, corr_w = graph_dict.get("corr", (None, None))
        emb_idx, emb_w = graph_dict.get("emb_sim", (None, None))

        sec_out = self._relation_forward(self.sector_gat, h, sec_idx, None)
        sec_out = F.elu(self.norm_sector(sec_out))

        corr_out = self._relation_forward(self.corr_gat, h, corr_idx, corr_w)
        corr_out = F.elu(self.norm_corr(corr_out))

        emb_out = self._relation_forward(self.emb_gat, h, emb_idx, emb_w)
        emb_out = F.elu(self.norm_emb(emb_out))

        alpha = torch.softmax(self.fusion_logits, dim=0)
        fused = alpha[0] * sec_out + alpha[1] * corr_out + alpha[2] * emb_out
        fused = self.dropout(fused)

        h_final = h + fused
        return h_final


class MultiRelationalGAT(nn.Module):
    """
    Multi-layer multi-relational GAT stack.
    """

    def __init__(self, embed_dim: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [_RelationalGATLayer(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, h: torch.Tensor, graph_dict: dict) -> torch.Tensor:
        out = h
        for layer in self.layers:
            out = layer(out, graph_dict)
        return out
