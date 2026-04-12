from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EPS = 1e-8


def information_coefficient(pred, actual):
    """Pearson correlation across stocks for one date."""
    p = np.asarray(pred, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    if p.size < 2 or np.nanstd(p) < EPS or np.nanstd(a) < EPS:
        return np.nan
    return float(np.corrcoef(p, a)[0, 1])


def rank_ic(pred, actual):
    """Spearman rank correlation across stocks for one date."""
    p = np.asarray(pred, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    if p.size < 2:
        return np.nan
    if np.nanstd(p) < EPS or np.nanstd(a) < EPS:
        return np.nan
    corr, _ = spearmanr(p, a, nan_policy="omit")
    return float(corr) if corr == corr else np.nan


def ic_ir(ic_series):
    """IC information ratio over time."""
    s = pd.Series(ic_series).dropna()
    if s.empty or s.std() < EPS:
        return np.nan
    return float(s.mean() / (s.std() + EPS))


def top_bottom_spread(pred, actual, k: int = 10):
    """Top-k minus bottom-k realized return spread."""
    p = np.asarray(pred, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.nan
    k = max(1, min(k, n // 2))
    order = np.argsort(p)
    bottom = order[:k]
    top = order[-k:]
    return float(np.nanmean(a[top]) - np.nanmean(a[bottom]))


def _cross_sectional_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _z(s: pd.Series) -> pd.Series:
        std = float(s.std(ddof=0))
        if std < EPS:
            return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
        return (s - s.mean()) / (std + EPS)

    return series.groupby(df["date"]).transform(_z)


def apply_alpha_score(
    predictions_df: pd.DataFrame,
    weights: tuple[float, float, float] | list[float],
    out_col: str = "alpha_score",
    use_zscore: bool = True,
    direction_component: str = "up_minus_down",
) -> pd.DataFrame:
    df = predictions_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if "pred_dir_score" not in df.columns:
        if direction_component == "up_prob" and "pred_dir_prob_up" in df.columns:
            df["pred_dir_score"] = df["pred_dir_prob_up"].astype(float)
        elif {"pred_dir_prob_down", "pred_dir_prob_up"}.issubset(df.columns):
            df["pred_dir_score"] = df["pred_dir_prob_up"].astype(float) - df["pred_dir_prob_down"].astype(float)
        else:
            df["pred_dir_score"] = df["pred_dir"].map({0: -1.0, 1: 0.0, 2: 1.0}).fillna(0.0).astype(float)

    wr, wk, wd = [float(x) for x in weights]
    if use_zscore:
        ret_comp = _cross_sectional_zscore(df, "pred_return")
        rank_comp = _cross_sectional_zscore(df, "pred_rank")
        dir_comp = _cross_sectional_zscore(df, "pred_dir_score")
    else:
        ret_comp = pd.to_numeric(df["pred_return"], errors="coerce").fillna(0.0)
        rank_comp = pd.to_numeric(df["pred_rank"], errors="coerce").fillna(0.0)
        dir_comp = pd.to_numeric(df["pred_dir_score"], errors="coerce").fillna(0.0)

    df[out_col] = wr * ret_comp + wk * rank_comp + wd * dir_comp
    return df


def select_alpha_configuration(predictions_df: pd.DataFrame, config) -> dict:
    if predictions_df.empty:
        return {
            "weights": (1.0, 0.0, 0.0),
            "score_col": "pred_return",
            "metrics": {},
            "predictions_df": predictions_df,
        }

    enabled = bool(getattr(config.evaluation, "alpha_search_enabled", False))
    if not enabled:
        score_col = str(getattr(config.evaluation, "portfolio_score_col", "pred_return"))
        return {
            "weights": (1.0, 0.0, 0.0),
            "score_col": score_col,
            "metrics": {},
            "predictions_df": predictions_df,
        }

    candidates = list(getattr(config.evaluation, "alpha_weight_candidates", []))
    if not candidates:
        candidates = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]

    use_zscore = bool(getattr(config.evaluation, "alpha_use_zscore", True))
    direction_component = str(getattr(config.evaluation, "alpha_direction_component", "up_minus_down"))

    best = None
    for weights in candidates:
        cand_df = apply_alpha_score(
            predictions_df,
            weights=weights,
            out_col="alpha_score",
            use_zscore=use_zscore,
            direction_component=direction_component,
        )
        cfg = copy.deepcopy(config)
        cfg.evaluation.portfolio_score_col = "alpha_score"
        metrics = compute_all_metrics(cand_df, cfg)["overall"]
        key = (
            float(metrics.get("sharpe_ratio", -np.inf)),
            float(metrics.get("score_rank_ic_mean", -np.inf)),
            float(metrics.get("spread_mean", -np.inf)),
            float(metrics.get("annualized_return", -np.inf)),
            -float(metrics.get("annualized_turnover", np.inf)),
            -abs(float(metrics.get("max_drawdown", np.inf))),
        )
        if best is None or key > best[0]:
            best = (
                key,
                tuple(float(x) for x in weights),
                metrics,
                cand_df,
            )

    assert best is not None
    return {
        "weights": best[1],
        "score_col": "alpha_score",
        "metrics": best[2],
        "predictions_df": best[3],
    }


def _labels_from_returns(returns, mode: str = "binary", threshold: float = 0.0):
    r = np.asarray(returns, dtype=np.float64)
    mode = str(mode).lower()
    if mode in {"binary", "sign"}:
        return (r > 0.0).astype(int)
    if mode in {"ternary", "ternary_threshold", "3class"}:
        return np.where(r > threshold, 2, np.where(r < -threshold, 0, 1)).astype(int)
    return (r > 0.0).astype(int)


def directional_accuracy(pred_dir, actual_dir):
    """Fraction where predicted direction class matches actual class."""
    pred_arr = np.asarray(pred_dir)
    if pred_arr.ndim == 2:
        pred = pred_arr.argmax(axis=1)
    else:
        pred = pred_arr.astype(int)
    actual = np.asarray(actual_dir)
    if actual.size == 0:
        return np.nan

    mask = np.isfinite(actual)
    if mask.sum() == 0:
        return np.nan
    return float((pred[mask] == actual[mask].astype(int)).mean())


def top_k_precision(pred, actual, k: int = 10):
    p = np.asarray(pred, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.nan
    k = max(1, min(k, n))
    order = np.argsort(p)
    top = order[-k:]
    return float(np.nanmean(a[top] > 0.0))


def _drawdown_stats(returns: pd.Series):
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = equity / (peak + EPS) - 1.0
    return dd, float(dd.min())


def _weights_from_scores(scores: pd.Series, top_k: int, long_short: bool) -> pd.Series:
    n = len(scores)
    if n == 0:
        return pd.Series(dtype=float)

    k = max(1, min(top_k, n // 2 if long_short else n))
    rank = np.argsort(scores.to_numpy(dtype=np.float64))

    w = np.zeros(n, dtype=np.float64)
    long_idx = rank[-k:]
    w[long_idx] = 1.0 / k
    if long_short:
        short_idx = rank[:k]
        w[short_idx] = -1.0 / k

    return pd.Series(w, index=scores.index, dtype=np.float64)


def _rebalance_bucket(ts: pd.Timestamp, rebalance_freq: str | None):
    if rebalance_freq is None:
        return ts
    freq = str(rebalance_freq).strip().upper()
    if freq in {"", "D", "DAILY", "1D"}:
        return ts
    try:
        return ts.to_period(rebalance_freq).start_time
    except Exception:
        return ts


def simulate_portfolio(
    all_dates,
    all_predictions,
    all_actual_returns,
    all_tickers=None,
    top_k: int = 10,
    long_short: bool = True,
    cost_bps: float = 10,
    rebalance_freq: str | None = "W-FRI",
    score_ema_alpha: float = 1.0,
) -> dict:
    """
    Simulate daily long-short or long-only portfolio from cross-sectional scores.
    """
    dates = pd.to_datetime(all_dates)
    order = np.argsort(dates.values)
    dates = dates[order]

    preds = [np.asarray(all_predictions[i], dtype=np.float64) for i in order]
    acts = [np.asarray(all_actual_returns[i], dtype=np.float64) for i in order]
    if all_tickers is not None:
        tics = [list(all_tickers[i]) for i in order]
    else:
        tics = [list(range(len(preds[i]))) for i in range(len(preds))]

    rets = []
    turnover_series = []
    prev_w: pd.Series | None = None
    prev_score: pd.Series | None = None
    prev_bucket = None
    alpha = float(np.clip(score_ema_alpha, 0.0, 1.0))

    for date, p, a, tk in zip(dates, preds, acts, tics):
        raw_scores = pd.Series(p, index=tk, dtype=np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if prev_score is None:
            score_s = raw_scores
        else:
            cur, prev = raw_scores.align(prev_score, join="left", fill_value=0.0)
            score_s = alpha * cur + (1.0 - alpha) * prev

        bucket = _rebalance_bucket(pd.Timestamp(date), rebalance_freq)
        do_rebalance = prev_w is None or bucket != prev_bucket

        if do_rebalance:
            w_s = _weights_from_scores(score_s, top_k=top_k, long_short=long_short)
        else:
            assert prev_w is not None
            w_s = prev_w.reindex(score_s.index).fillna(0.0)

        a_s = pd.Series(a, index=tk, dtype=np.float64).reindex(w_s.index).fillna(0.0)

        gross = float((w_s * a_s).sum())
        if prev_w is not None:
            cur, prev = w_s.align(prev_w, join="outer", fill_value=0.0)
            turnover = float((cur - prev).abs().sum())
        else:
            turnover = float(w_s.abs().sum())

        cost = (cost_bps / 10000.0) * turnover
        net = gross - cost

        rets.append(net)
        turnover_series.append(turnover)
        prev_w = w_s
        prev_score = score_s
        prev_bucket = bucket

    returns = pd.Series(rets, index=dates, name="strategy_return")
    turnover_s = pd.Series(turnover_series, index=dates, name="turnover")

    ann_factor = 252
    mean_r = returns.mean()
    std_r = returns.std()
    downside_std = returns[returns < 0].std()

    ann_return = float((1.0 + returns).prod() ** (ann_factor / max(len(returns), 1)) - 1.0)
    sharpe = float((mean_r / (std_r + EPS)) * np.sqrt(ann_factor))
    sortino = float((mean_r / (downside_std + EPS)) * np.sqrt(ann_factor))
    dd, max_dd = _drawdown_stats(returns)
    hit_ratio = float((returns > 0).mean())
    ann_turnover = float(turnover_s.mean() * ann_factor)
    calmar = float(ann_return / (abs(max_dd) + EPS))

    metrics = {
        "annualized_return": ann_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "hit_ratio": hit_ratio,
        "annualized_turnover": ann_turnover,
        "calmar_ratio": calmar,
    }
    return {
        "returns": returns,
        "turnover": turnover_s,
        "drawdown": dd,
        "metrics": metrics,
    }


def compute_all_metrics(predictions_df: pd.DataFrame, config) -> dict:
    """
    Compute cross-sectional and portfolio metrics from prediction rows.
    """
    if predictions_df.empty:
        return {}

    df = predictions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"])

    score_col = str(getattr(config.evaluation, "portfolio_score_col", "pred_return"))
    if score_col not in df.columns:
        score_col = "pred_return"

    direction_mode = str(getattr(config.features, "direction_label_mode", "binary"))
    direction_threshold = float(getattr(config.features, "direction_threshold", 0.0))

    daily_rows = []
    dates, daily_preds, daily_actuals, daily_tickers = [], [], [], []
    for date, g in df.groupby("date"):
        pred_ret = g["pred_return"].to_numpy(dtype=float)
        pred_rank = g["pred_rank"].to_numpy(dtype=float) if "pred_rank" in g else pred_ret
        portfolio_score = g[score_col].to_numpy(dtype=float)
        actual = g["actual_return"].to_numpy(dtype=float)
        pred_dir = g["pred_dir"].to_numpy(dtype=float) if "pred_dir" in g else _labels_from_returns(pred_ret, mode=direction_mode, threshold=direction_threshold)
        if "actual_dir" in g.columns:
            actual_dir = g["actual_dir"].to_numpy(dtype=float)
        else:
            actual_dir = _labels_from_returns(actual, mode=direction_mode, threshold=direction_threshold)

        d_ic = information_coefficient(pred_ret, actual)
        d_ric = rank_ic(pred_rank, actual)
        d_ric_ret = rank_ic(pred_ret, actual)
        d_score_ric = rank_ic(portfolio_score, actual)
        d_spread = top_bottom_spread(portfolio_score, actual, k=int(config.evaluation.top_k))
        d_dacc = directional_accuracy(pred_dir, actual_dir)
        d_topk_prec = top_k_precision(portfolio_score, actual, k=int(config.evaluation.top_k))

        daily_rows.append(
            {
                "date": date,
                "ic": d_ic,
                "rank_ic": d_ric,
                "rank_ic_ret": d_ric_ret,
                "score_rank_ic": d_score_ric,
                "spread": d_spread,
                "directional_accuracy": d_dacc,
                "top_k_precision": d_topk_prec,
            }
        )
        dates.append(date)
        daily_preds.append(portfolio_score)
        daily_actuals.append(actual)
        daily_tickers.append(g["ticker"].astype(str).tolist())

    daily_df = pd.DataFrame(daily_rows).set_index("date")

    sim = simulate_portfolio(
        all_dates=dates,
        all_predictions=daily_preds,
        all_actual_returns=daily_actuals,
        all_tickers=daily_tickers,
        top_k=int(config.evaluation.top_k),
        long_short=bool(config.evaluation.long_short),
        cost_bps=float(config.evaluation.transaction_cost_bps),
        rebalance_freq=getattr(config.evaluation, "rebalance_frequency", "W-FRI"),
        score_ema_alpha=float(getattr(config.evaluation, "score_ema_alpha", 1.0)),
    )

    out = {
        "ic_mean": float(daily_df["ic"].mean()),
        "ic_std": float(daily_df["ic"].std()),
        "ic_ir": float(ic_ir(daily_df["ic"])),
        "rank_ic_mean": float(daily_df["rank_ic"].mean()),
        "rank_ic_std": float(daily_df["rank_ic"].std()),
        "rank_ic_ret_mean": float(daily_df["rank_ic_ret"].mean()),
        "rank_ic_ret_std": float(daily_df["rank_ic_ret"].std()),
        "score_rank_ic_mean": float(daily_df["score_rank_ic"].mean()),
        "score_rank_ic_std": float(daily_df["score_rank_ic"].std()),
        "spread_mean": float(daily_df["spread"].mean()),
        "directional_accuracy": float(daily_df["directional_accuracy"].mean()),
        "top_k_precision": float(daily_df["top_k_precision"].mean()),
    }
    out.update(sim["metrics"])

    return {
        "overall": out,
        "daily": daily_df,
        "portfolio_returns": sim["returns"],
        "portfolio_turnover": sim["turnover"],
        "portfolio_drawdown": sim["drawdown"],
    }
