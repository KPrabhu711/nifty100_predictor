from __future__ import annotations

import pandas as pd
from rich.console import Console
from rich.table import Table


def _slice_dates(all_dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    mask = (all_dates >= start) & (all_dates <= end)
    return [d.strftime("%Y-%m-%d") for d in all_dates[mask]]


def generate_walkforward_splits(all_dates, config) -> list[dict]:
    """
    Generates walk-forward folds with train/val/test date lists.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(sorted(all_dates)))
    if len(dates) == 0:
        return []

    train_years = int(config.walkforward.train_years)
    val_years = int(config.walkforward.val_years)
    test_years = int(config.walkforward.test_years)
    step_months = int(config.walkforward.step_months)
    mode = str(getattr(config.walkforward, "mode", "rolling")).lower()

    start = dates.min()
    base_train_end = start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)

    folds: list[dict] = []
    offset = 0
    while True:
        shift = pd.DateOffset(months=offset)
        train_start = start if mode == "expanding" else start + shift
        train_end = base_train_end + shift
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.DateOffset(years=val_years) - pd.Timedelta(days=1)
        test_start = val_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)

        if test_start > dates.max():
            break

        train_dates = _slice_dates(dates, train_start, train_end)
        val_dates = _slice_dates(dates, val_start, val_end)
        test_dates = _slice_dates(dates, test_start, test_end)

        if not train_dates or not val_dates or not test_dates:
            break

        folds.append(
            {
                "fold": len(folds),
                "train_dates": train_dates,
                "val_dates": val_dates,
                "test_dates": test_dates,
            }
        )
        offset += step_months

    table = Table(title="Walk-forward Splits")
    table.add_column("Fold", justify="right")
    table.add_column("Train")
    table.add_column("Val")
    table.add_column("Test")
    table.add_column("Counts", justify="right")

    for split in folds:
        tr = split["train_dates"]
        va = split["val_dates"]
        te = split["test_dates"]
        counts = f"{len(tr)}/{len(va)}/{len(te)}"
        table.add_row(
            str(split["fold"]),
            f"{tr[0]} -> {tr[-1]}",
            f"{va[0]} -> {va[-1]}",
            f"{te[0]} -> {te[-1]}",
            counts,
        )

    Console().print(table)
    return folds
