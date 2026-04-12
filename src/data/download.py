from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
import time

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn


@dataclass
class DownloadSummary:
    requested: int
    downloaded: int
    skipped_fresh: int
    missing: int


TICKER_ALIASES: dict[str, list[str]] = {
    # Infosys Yahoo symbol is INFY.NS; universe keeps INFOSYS.NS by design.
    "INFOSYS.NS": ["INFY.NS"],
    # Zomato was renamed on exchanges; keep compatibility fallback.
    "ZOMATO.NS": ["ETERNAL.NS"],
    # Legacy alias for United Spirits on Yahoo.
    "MCDOWELL-N.NS": ["UNITDSPR.NS"],
}


def _ensure_datetime(date_like: str | datetime) -> datetime:
    if isinstance(date_like, datetime):
        return date_like
    return datetime.strptime(str(date_like), "%Y-%m-%d")


def _is_fresh(parquet_path: Path, end_date: datetime, freshness_buffer_days: int = 7) -> bool:
    if not parquet_path.exists():
        return False
    try:
        df = pd.read_parquet(parquet_path, columns=["Date"])
        if df.empty:
            return False
        max_date = pd.to_datetime(df["Date"]).max()
        target = pd.Timestamp(end_date - timedelta(days=freshness_buffer_days))
        return bool(max_date >= target)
    except Exception:
        return False


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out.rename(columns={c: c.strip() for c in out.columns})
    keep_cols = ["Open", "High", "Low", "Close", "Volume"]
    existing = [c for c in keep_cols if c in out.columns]
    out = out[existing].reset_index().rename(columns={"index": "Date"})
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date")
    return out


def _download_market_symbol(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(0):
            df = df[symbol]
        else:
            df = df.droplevel(0, axis=1)
    return _clean_ohlcv(df)


def _download_single_symbol(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(0):
            df = df[symbol]
        else:
            df = df.droplevel(0, axis=1)
    return _clean_ohlcv(df.dropna(how="all"))


def _download_symbol_via_ticker_history(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)

    # Primary attempt with bounded date range.
    hist = ticker.history(
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        actions=False,
    )

    # Yahoo occasionally returns empty for bounded windows; fallback to max period then clip.
    if hist is None or hist.empty:
        hist = ticker.history(
            period="max",
            interval="1d",
            auto_adjust=True,
            actions=False,
        )

    if hist is None or hist.empty:
        return pd.DataFrame()

    hist = hist.copy()
    hist.index = pd.to_datetime(hist.index)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    hist = hist[(hist.index >= start_ts) & (hist.index <= end_ts)]
    return _clean_ohlcv(hist.dropna(how="all"))


def _download_with_fallbacks(ticker: str, start_date: str, end_date: str, logger=None) -> tuple[pd.DataFrame, str | None]:
    candidates = [ticker] + TICKER_ALIASES.get(ticker, [])

    for symbol in candidates:
        for attempt in range(3):
            try:
                tdf = _download_single_symbol(symbol=symbol, start_date=start_date, end_date=end_date)
                if not tdf.empty:
                    return tdf, symbol
            except Exception as exc:
                if logger:
                    logger.warning(
                        "single_retry_failed ticker=%s symbol=%s attempt=%d err=%s",
                        ticker,
                        symbol,
                        attempt + 1,
                        str(exc),
                    )

            try:
                tdf_hist = _download_symbol_via_ticker_history(symbol=symbol, start_date=start_date, end_date=end_date)
                if not tdf_hist.empty:
                    return tdf_hist, symbol
            except Exception as exc:
                if logger:
                    logger.warning(
                        "ticker_history_retry_failed ticker=%s symbol=%s attempt=%d err=%s",
                        ticker,
                        symbol,
                        attempt + 1,
                        str(exc),
                    )

            time.sleep(0.5)

    return pd.DataFrame(), None


def download_all(config, logger=None) -> DownloadSummary:
    console = Console()
    raw_dir = Path(config.data.raw_dir)
    universe_file = Path(config.data.universe_file)
    raw_dir.mkdir(parents=True, exist_ok=True)

    universe = pd.read_csv(universe_file)
    tickers = universe["ticker"].dropna().astype(str).tolist()

    start_date = str(config.data.start_date)
    end_date = str(config.data.end_date)
    end_dt = _ensure_datetime(end_date)

    fresh, stale = [], []
    for ticker in tickers:
        p = raw_dir / f"{ticker}.parquet"
        if _is_fresh(p, end_dt):
            fresh.append(ticker)
        else:
            stale.append(ticker)

    if logger:
        logger.info(
            "download_start requested=%d stale=%d fresh=%d range=%s:%s",
            len(tickers),
            len(stale),
            len(fresh),
            start_date,
            end_date,
        )

    downloaded = 0
    missing = 0

    fallback_candidates: list[str] = []

    if stale:
        batch = yf.download(
            tickers=stale,
            start=start_date,
            end=end_date,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        with Progress(
            TextColumn("[bold cyan]Saving ticker parquet files"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("save", total=len(stale))
            for ticker in stale:
                try:
                    if isinstance(batch.columns, pd.MultiIndex):
                        if ticker not in batch.columns.get_level_values(0):
                            fallback_candidates.append(ticker)
                            if logger:
                                logger.warning("missing_ticker_data_batch ticker=%s scheduling_single_retry", ticker)
                            progress.advance(task)
                            continue
                        tdf = batch[ticker]
                    else:
                        # Single-column responses from batch are unreliable for multi-ticker requests.
                        fallback_candidates.append(ticker)
                        progress.advance(task)
                        continue

                    tdf = _clean_ohlcv(tdf.dropna(how="all"))
                    if tdf.empty:
                        fallback_candidates.append(ticker)
                        if logger:
                            logger.warning("empty_ticker_data_batch ticker=%s scheduling_single_retry", ticker)
                        progress.advance(task)
                        continue

                    tdf.to_parquet(raw_dir / f"{ticker}.parquet", index=False)
                    downloaded += 1
                except Exception as exc:
                    fallback_candidates.append(ticker)
                    if logger:
                        logger.warning("download_failed_batch ticker=%s err=%s scheduling_single_retry", ticker, str(exc))
                progress.advance(task)

    if fallback_candidates:
        unique_retry = sorted(set(fallback_candidates))
        if logger:
            logger.info("single_retry_start count=%d", len(unique_retry))

        with Progress(
            TextColumn("[bold yellow]Retrying failed tickers"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("retry", total=len(unique_retry))
            for ticker in unique_retry:
                try:
                    tdf, used_symbol = _download_with_fallbacks(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        logger=logger,
                    )
                    if tdf.empty:
                        missing += 1
                        if logger:
                            logger.warning("single_retry_failed_final ticker=%s", ticker)
                    else:
                        tdf.to_parquet(raw_dir / f"{ticker}.parquet", index=False)
                        downloaded += 1
                        if logger and used_symbol != ticker:
                            logger.info("alias_symbol_used ticker=%s symbol=%s", ticker, used_symbol)
                except Exception as exc:
                    missing += 1
                    if logger:
                        logger.warning("single_retry_exception ticker=%s err=%s", ticker, str(exc))
                progress.advance(task)

    market_targets = {
        "NIFTY100_INDEX": "^CNX100",
        "INDIA_VIX": "^INDIAVIX",
    }
    for out_name, symbol in market_targets.items():
        out_path = raw_dir / f"{out_name}.parquet"
        if _is_fresh(out_path, end_dt):
            continue
        try:
            mdf = _download_market_symbol(symbol, start_date, end_date)
            if not mdf.empty:
                mdf.to_parquet(out_path, index=False)
                if logger:
                    logger.info("downloaded_market_symbol name=%s symbol=%s rows=%d", out_name, symbol, len(mdf))
            elif logger:
                logger.warning("empty_market_symbol name=%s symbol=%s", out_name, symbol)
        except Exception as exc:
            if logger:
                logger.warning("market_symbol_failed name=%s symbol=%s err=%s", out_name, symbol, str(exc))

    if logger:
        logger.info(
            "download_complete success=%d missing=%d skipped_fresh=%d total=%d",
            downloaded,
            missing,
            len(fresh),
            len(tickers),
        )

    return DownloadSummary(
        requested=len(tickers),
        downloaded=downloaded,
        skipped_fresh=len(fresh),
        missing=missing,
    )
