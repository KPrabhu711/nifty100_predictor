from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


plt.style.use("dark_background")
PRIMARY = "#00D4FF"
SECONDARY = "#FF6B35"
ACCENT = "#7FFF00"


def _ensure_parent(save_path: str | Path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig, save_path: str | Path):
    path = _ensure_parent(save_path)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pretrain_loss(loss_history: list, save_path: str | Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_history, color=PRIMARY, linewidth=2)
    ax.set_title("Pretrain Reconstruction Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(alpha=0.2)
    _save(fig, save_path)


def plot_train_val_loss(train_losses, val_losses, fold_id, save_path: str | Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, color=PRIMARY, linewidth=2, label="Train")
    ax.plot(val_losses, color=SECONDARY, linewidth=2, label="Val")
    ax.set_title(f"Fold {fold_id}: Train vs Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, save_path)


def plot_ic_over_time(ic_series: pd.Series, rank_ic_series: pd.Series, save_path: str | Path):
    ic = pd.Series(ic_series).dropna()
    ric = pd.Series(rank_ic_series).dropna()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ic.index, ic.values, color=PRIMARY, alpha=0.5, label="IC")
    ax.plot(ric.index, ric.values, color=SECONDARY, alpha=0.5, label="Rank IC")
    ax.plot(ic.rolling(20).mean(), color=PRIMARY, linewidth=2.5, label="IC 20D MA")
    ax.plot(ric.rolling(20).mean(), color=SECONDARY, linewidth=2.5, label="Rank IC 20D MA")
    ax.axhline(0.0, color="white", linestyle="--", linewidth=1)
    ax.fill_between(ic.index, 0, ic.values, where=ic.values > 0, color=ACCENT, alpha=0.15)
    ax.set_title("IC and Rank IC Over Time")
    ax.set_ylabel("Correlation")
    ax.legend(ncol=2)
    ax.grid(alpha=0.2)
    _save(fig, save_path)


def plot_ic_distribution(ic_values: np.ndarray, save_path: str | Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(ic_values, bins=30, kde=True, color=PRIMARY, ax=ax)
    ax.axvline(np.nanmean(ic_values), color=SECONDARY, linestyle="--", label="Mean IC")
    ax.set_title("IC Distribution")
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, save_path)


def plot_portfolio_cumulative_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    save_path: str | Path,
):
    port = pd.Series(portfolio_returns).fillna(0.0)
    bench = pd.Series(benchmark_returns).reindex(port.index).fillna(0.0)

    port_cum = (1.0 + port).cumprod()
    bench_cum = (1.0 + bench).cumprod()
    peak = port_cum.cummax()
    dd = port_cum / peak - 1.0
    max_dd_date = dd.idxmin() if len(dd) else None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, color=PRIMARY, linewidth=2, label="Strategy")
    ax.plot(bench_cum.index, bench_cum.values, color=SECONDARY, linewidth=2, label="NIFTY100")
    ax.fill_between(port_cum.index, port_cum.values, peak.values, where=dd.values < 0, color=SECONDARY, alpha=0.1)

    if max_dd_date is not None:
        ax.annotate(
            f"Max DD: {dd.loc[max_dd_date]:.2%}",
            xy=(max_dd_date, port_cum.loc[max_dd_date]),
            xytext=(20, -20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=ACCENT),
            color=ACCENT,
        )

    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, save_path)


def plot_drawdown(portfolio_returns: pd.Series, save_path: str | Path):
    ret = pd.Series(portfolio_returns).fillna(0.0)
    equity = (1.0 + ret).cumprod()
    dd = equity / equity.cummax() - 1.0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0.0, color=SECONDARY, alpha=0.35)
    ax.plot(dd.index, dd.values, color=SECONDARY, linewidth=1.5)
    ax.set_title("Underwater Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.2)
    _save(fig, save_path)


def plot_top_bottom_spread(spread_series: pd.Series, save_path: str | Path):
    s = pd.Series(spread_series)
    if isinstance(s.index, pd.DatetimeIndex):
        try:
            monthly = s.resample("ME").mean()
        except ValueError:
            monthly = s.resample("M").mean()
    else:
        monthly = s

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [ACCENT if v >= 0 else SECONDARY for v in monthly.values]
    ax.bar(monthly.index.astype(str), monthly.values, color=colors, alpha=0.85)
    ax.axhline(0.0, color="white", linestyle="--", linewidth=1)
    ax.set_title("Monthly Top-Bottom Spread")
    ax.set_ylabel("Return Spread")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.2, axis="y")
    _save(fig, save_path)


def plot_loss_components(reg_losses, rank_losses, dir_losses, fold_id, save_path: str | Path):
    x = np.arange(len(reg_losses))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        x,
        reg_losses,
        rank_losses,
        dir_losses,
        labels=["Regression", "Ranking", "Direction"],
        colors=[PRIMARY, SECONDARY, ACCENT],
        alpha=0.8,
    )
    ax.set_title(f"Fold {fold_id}: Loss Components")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    _save(fig, save_path)


def plot_feature_importance(feature_names, importance_scores, save_path: str | Path):
    df = pd.DataFrame({"feature": feature_names, "score": importance_scores}).dropna()
    top = df.sort_values("score", ascending=False).head(30).sort_values("score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(top["feature"], top["score"], color=PRIMARY, alpha=0.85)
    ax.set_title("Top 30 Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.grid(alpha=0.2, axis="x")
    _save(fig, save_path)


def plot_sector_ic(sector_ic_dict: dict, save_path: str | Path):
    s = pd.Series(sector_ic_dict).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(s.index, s.values, color=PRIMARY, alpha=0.85)
    ax.axhline(0.0, color="white", linestyle="--", linewidth=1)
    ax.set_title("Mean IC by Sector")
    ax.set_ylabel("Mean IC")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.2, axis="y")
    _save(fig, save_path)


def plot_walkforward_summary(fold_metrics: list[dict], save_path: str | Path):
    df = pd.DataFrame(fold_metrics)
    folds = df.get("fold", pd.Series(range(len(df))))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    axes[0].plot(folds, df.get("sharpe_ratio", pd.Series([np.nan] * len(df))), color=PRIMARY, marker="o")
    axes[0].set_title("Sharpe by Fold")

    axes[1].plot(folds, df.get("rank_ic_mean", pd.Series([np.nan] * len(df))), color=SECONDARY, marker="o")
    axes[1].set_title("Rank IC by Fold")

    axes[2].plot(folds, df.get("max_drawdown", pd.Series([np.nan] * len(df))), color=ACCENT, marker="o")
    axes[2].set_title("Max Drawdown by Fold")

    axes[3].plot(folds, df.get("annualized_return", pd.Series([np.nan] * len(df))), color=PRIMARY, marker="o")
    axes[3].set_title("Annualized Return by Fold")

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlabel("Fold")

    _save(fig, save_path)


def plot_alpha_weight_summary(fold_metrics: list[dict], save_path: str | Path):
    df = pd.DataFrame(fold_metrics)
    needed = {"alpha_w_ret", "alpha_w_rank", "alpha_w_dir"}
    if not needed.issubset(df.columns):
        return

    folds = df.get("fold", pd.Series(range(len(df))))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(folds, df["alpha_w_ret"], color=PRIMARY, marker="o", label="Return")
    ax.plot(folds, df["alpha_w_rank"], color=SECONDARY, marker="o", label="Rank")
    ax.plot(folds, df["alpha_w_dir"], color=ACCENT, marker="o", label="Direction")
    ax.set_title("Selected Alpha Weights by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend()
    _save(fig, save_path)


def plot_embedding_similarity_heatmap(
    embeddings: np.ndarray,
    tickers: list[str],
    save_path: str | Path,
):
    emb = np.asarray(embeddings, dtype=np.float64)
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    sim = np.clip(sim, -1.0, 1.0)

    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)

    sim_ord = sim[order][:, order]
    tick_ord = [tickers[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sim_ord, cmap="mako", ax=ax, xticklabels=tick_ord, yticklabels=tick_ord, cbar=True)
    ax.set_title("Embedding Cosine Similarity (Clustered)")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelrotation=0, labelsize=6)

    sector_map_path = Path("data/processed/sector_map.parquet")
    if sector_map_path.exists():
        try:
            sec = pd.read_parquet(sector_map_path)
            sec_map = dict(zip(sec["ticker"], sec["sector"]))
            sec_order = [sec_map.get(t, "Unknown") for t in tick_ord]
            uniq = sorted(set(sec_order))
            palette = sns.color_palette("tab20", n_colors=len(uniq))
            c_map = {s: palette[i] for i, s in enumerate(uniq)}
            colors = np.array([c_map[s] for s in sec_order])

            top_ax = ax.inset_axes([0.0, 1.01, 1.0, 0.03])
            top_ax.imshow(colors[np.newaxis, :, :], aspect="auto")
            top_ax.axis("off")

            left_ax = ax.inset_axes([-0.035, 0.0, 0.03, 1.0])
            left_ax.imshow(colors[:, np.newaxis, :], aspect="auto")
            left_ax.axis("off")
        except Exception:
            pass

    _save(fig, save_path)


def generate_full_report(all_metrics, config, save_dir: str | Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(config.logging.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = all_metrics.get("fold_metrics", [])
    if fold_metrics:
        summary_df = pd.DataFrame(fold_metrics)
        overall_row = all_metrics.get("overall", {})
        if isinstance(overall_row, dict) and overall_row:
            overall_row = dict(overall_row)
            overall_row.setdefault("fold", "ALL")
            summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)
        summary_df.to_csv(metrics_dir / "final_summary.csv", index=False)
        plot_walkforward_summary(fold_metrics, save_dir / "walkforward_summary.png")
        plot_alpha_weight_summary(fold_metrics, save_dir / "alpha_weight_summary.png")

    if "ic_series" in all_metrics and "rank_ic_series" in all_metrics:
        plot_ic_over_time(
            all_metrics["ic_series"],
            all_metrics["rank_ic_series"],
            save_dir / "ic_over_time.png",
        )
        plot_ic_distribution(np.asarray(all_metrics["ic_series"]), save_dir / "ic_distribution.png")

    if "portfolio_returns" in all_metrics and "benchmark_returns" in all_metrics:
        plot_portfolio_cumulative_returns(
            all_metrics["portfolio_returns"],
            all_metrics["benchmark_returns"],
            save_dir / "portfolio_cumulative_returns.png",
        )
        plot_drawdown(all_metrics["portfolio_returns"], save_dir / "portfolio_drawdown.png")

    if "spread_series" in all_metrics:
        plot_top_bottom_spread(all_metrics["spread_series"], save_dir / "top_bottom_spread.png")

    if "feature_importance" in all_metrics:
        fi = all_metrics["feature_importance"]
        plot_feature_importance(fi["names"], fi["scores"], save_dir / "feature_importance.png")

    if "sector_ic" in all_metrics:
        plot_sector_ic(all_metrics["sector_ic"], save_dir / "sector_ic.png")

    if "embeddings" in all_metrics and "tickers" in all_metrics:
        plot_embedding_similarity_heatmap(
            all_metrics["embeddings"],
            all_metrics["tickers"],
            save_dir / "embedding_similarity_heatmap.png",
        )
