from __future__ import annotations

import torch


EPS = 1e-8


def _empty_edges(device: torch.device):
    return torch.empty((2, 0), dtype=torch.long, device=device)


def build_sector_graph(sector_ids: torch.LongTensor, n_stocks: int) -> torch.Tensor:
    """
    Fully connects all stocks within the same sector.
    Returns edge_index: [2, E_sector]
    """
    device = sector_ids.device
    edges = []
    unique = torch.unique(sector_ids)
    for sid in unique:
        nodes = torch.where(sector_ids == sid)[0]
        if nodes.numel() <= 1:
            continue
        src = nodes.repeat_interleave(nodes.numel())
        dst = nodes.repeat(nodes.numel())
        mask = src != dst
        src, dst = src[mask], dst[mask]
        if src.numel() > 0:
            edges.append(torch.stack([src, dst], dim=0))

    if not edges:
        return _empty_edges(device)
    return torch.cat(edges, dim=1)


def build_rolling_corr_graph(
    returns: torch.Tensor,
    threshold: float = 0.4,
    top_k_per_node: int = 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pairwise Pearson correlation graph over rolling window returns.
    """
    device = returns.device
    n, w = returns.shape
    if n == 0 or w < 2:
        return _empty_edges(device), torch.empty(
            (0,), dtype=torch.float32, device=device
        )

    x = returns.float()
    x = x - x.mean(dim=1, keepdim=True)
    cov = (x @ x.t()) / max(w - 1, 1)
    var = (x.pow(2).sum(dim=1) / max(w - 1, 1)).clamp_min(EPS)
    std = torch.sqrt(var)
    corr = cov / (std.unsqueeze(1) * std.unsqueeze(0) + EPS)
    corr = corr.clamp(-1.0, 1.0)

    edge_src = []
    edge_dst = []
    edge_w = []
    for i in range(n):
        row = corr[i].clone()
        row[i] = 0.0
        keep = torch.where(torch.abs(row) >= threshold)[0]
        if keep.numel() == 0:
            continue
        if keep.numel() > top_k_per_node:
            vals = torch.abs(row[keep])
            top_vals, top_idx = torch.topk(vals, k=top_k_per_node)
            keep = keep[top_idx]
        weights = row[keep]
        edge_src.append(torch.full((keep.numel(),), i, dtype=torch.long, device=device))
        edge_dst.append(keep.long())
        edge_w.append(weights.float())

    if not edge_src:
        return _empty_edges(device), torch.empty(
            (0,), dtype=torch.float32, device=device
        )

    src = torch.cat(edge_src)
    dst = torch.cat(edge_dst)
    ew = torch.cat(edge_w).clamp(-1.0, 1.0)
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, ew


def build_embedding_sim_graph(
    embeddings: torch.Tensor,
    threshold: float = 0.5,
    top_k_per_node: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cosine similarity graph from current stock embeddings.
    """
    device = embeddings.device
    n = embeddings.size(0)
    if n == 0:
        return _empty_edges(device), torch.empty(
            (0,), dtype=torch.float32, device=device
        )

    h = embeddings.float()
    h = h / (h.norm(dim=1, keepdim=True) + EPS)
    sim = (h @ h.t()).clamp(-1.0, 1.0)

    edge_src = []
    edge_dst = []
    edge_w = []
    for i in range(n):
        row = sim[i].clone()
        row[i] = -1.0
        keep = torch.where(row >= threshold)[0]
        if keep.numel() == 0:
            continue
        if keep.numel() > top_k_per_node:
            vals = row[keep]
            _, top_idx = torch.topk(vals, k=top_k_per_node)
            keep = keep[top_idx]
        weights = row[keep]
        edge_src.append(torch.full((keep.numel(),), i, dtype=torch.long, device=device))
        edge_dst.append(keep.long())
        edge_w.append(weights.float())

    if not edge_src:
        return _empty_edges(device), torch.empty(
            (0,), dtype=torch.float32, device=device
        )

    src = torch.cat(edge_src)
    dst = torch.cat(edge_dst)
    ew = torch.cat(edge_w)
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, ew


def build_combined_graph(sector_ids, embeddings, raw_returns, config) -> dict:
    """
    Returns dict with relation-specific edge_index/edge_weight pairs.
    """
    if embeddings is not None:
        device = embeddings.device
    elif raw_returns is not None:
        device = raw_returns.device
    else:
        device = sector_ids.device

    sector_ids = sector_ids.to(device)
    n_stocks = int(sector_ids.numel())
    relation_types = {
        str(r)
        for r in getattr(
            config.model, "relation_types", ["sector", "rolling_corr", "emb_similarity"]
        )
    }

    if "sector" in relation_types:
        sector_edges = build_sector_graph(sector_ids=sector_ids, n_stocks=n_stocks)
    else:
        sector_edges = _empty_edges(device)

    if "rolling_corr" in relation_types or "corr" in relation_types:
        corr_edges, corr_w = build_rolling_corr_graph(
            returns=raw_returns.to(device)
            if raw_returns is not None
            else torch.zeros((n_stocks, 1), device=device),
            threshold=float(config.model.corr_threshold),
            top_k_per_node=15,
        )
    else:
        corr_edges = _empty_edges(device)
        corr_w = torch.empty((0,), dtype=torch.float32, device=device)

    if (
        embeddings is None
        or "emb_similarity" not in relation_types
        and "emb_sim" not in relation_types
    ):
        emb_edges = _empty_edges(device)
        emb_w = torch.empty((0,), dtype=torch.float32, device=device)
    else:
        emb_edges, emb_w = build_embedding_sim_graph(
            embeddings=embeddings,
            threshold=float(config.model.emb_sim_threshold),
            top_k_per_node=10,
        )

    return {
        "sector": (sector_edges, None),
        "corr": (corr_edges, corr_w),
        "emb_sim": (emb_edges, emb_w),
    }
