from __future__ import annotations

import torch
import torch.nn as nn

from .gnn import MultiRelationalGAT
from .graph import build_combined_graph, build_embedding_sim_graph, build_rolling_corr_graph
from .heads import DirectionHead, RankingHead, ReturnRegressionHead
from .patchtst import MultiScalePatchTST


class NIFTY100PredictionModel(nn.Module):
    """
    Full NIFTY100 return prediction model.
    """

    def __init__(self, config, feature_dim: int):
        super().__init__()
        self.config = config
        mcfg = config.model

        self.temporal_encoder = MultiScalePatchTST(
            seq_len=int(config.features.lookback),
            in_dim=feature_dim,
            embed_dim=int(mcfg.embedding_dim),
            patch_sizes=list(mcfg.patch_sizes),
            patch_strides=list(mcfg.patch_strides),
            n_layers=int(mcfg.n_transformer_layers),
            n_heads=int(mcfg.n_attention_heads),
            dropout=float(mcfg.dropout),
            gradient_checkpointing=bool(config.training.gradient_checkpointing),
        )

        self.graph_encoder = MultiRelationalGAT(
            embed_dim=int(mcfg.embedding_dim),
            n_heads=int(mcfg.n_attention_heads),
            n_layers=int(mcfg.graph_layers),
            dropout=float(mcfg.dropout),
        )

        self.expected_regime_dim = int(mcfg.regime_dim)
        head_in_dim = int(mcfg.embedding_dim) + self.expected_regime_dim
        self.ret_head = ReturnRegressionHead(in_dim=head_in_dim, dropout=float(mcfg.dropout))
        self.rank_head = RankingHead(in_dim=head_in_dim, dropout=float(mcfg.dropout))
        self.direction_n_classes = int(getattr(mcfg, "direction_n_classes", 2))
        self.dir_head = DirectionHead(
            in_dim=head_in_dim,
            num_classes=self.direction_n_classes,
            dropout=float(mcfg.dropout),
        )

    def forward(
        self,
        features: torch.Tensor,
        regime: torch.Tensor,
        graph_dict: dict,
        raw_returns: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        temporal_emb = self.temporal_encoder(features)

        if graph_dict is None:
            raise ValueError("graph_dict is required and must include sector relation or sector_ids")

        if "sector_ids" in graph_dict:
            combined = build_combined_graph(
                sector_ids=graph_dict["sector_ids"],
                embeddings=temporal_emb,
                raw_returns=raw_returns,
                config=self.config,
            )
        else:
            combined = dict(graph_dict)
            corr_idx, corr_w = build_rolling_corr_graph(
                returns=raw_returns,
                threshold=float(self.config.model.corr_threshold),
                top_k_per_node=15,
            )
            emb_idx, emb_w = build_embedding_sim_graph(
                embeddings=temporal_emb,
                threshold=float(self.config.model.emb_sim_threshold),
                top_k_per_node=10,
            )
            combined["corr"] = (corr_idx, corr_w)
            combined["emb_sim"] = (emb_idx, emb_w)

        graph_emb = self.graph_encoder(temporal_emb, combined)

        if regime.dim() == 1:
            regime_expand = regime.unsqueeze(0).expand(graph_emb.size(0), -1)
        else:
            regime_expand = regime.expand(graph_emb.size(0), -1)

        regime_expand = regime_expand.to(dtype=graph_emb.dtype, device=graph_emb.device)
        current_dim = regime_expand.size(-1)
        if current_dim < self.expected_regime_dim:
            pad = torch.zeros(
                (graph_emb.size(0), self.expected_regime_dim - current_dim),
                dtype=regime_expand.dtype,
                device=regime_expand.device,
            )
            regime_expand = torch.cat([regime_expand, pad], dim=-1)
        elif current_dim > self.expected_regime_dim:
            regime_expand = regime_expand[:, : self.expected_regime_dim]

        final_emb = torch.cat([graph_emb, regime_expand], dim=-1)

        ret_pred = self.ret_head(final_emb)
        rank_score = self.rank_head(final_emb)
        dir_logits = self.dir_head(final_emb)

        return {
            "ret_pred": ret_pred,
            "rank_score": rank_score,
            "dir_logits": dir_logits,
            "embeddings": graph_emb,
        }
