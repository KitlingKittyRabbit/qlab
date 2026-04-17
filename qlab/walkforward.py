"""
Walk-forward cross-validation splitter.

Generates non-overlapping train/test date splits with optional embargo
to prevent forward-return leakage from training into test period.
"""

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass
class WalkForwardFold:
    """A single walk-forward fold."""
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_days: int,
    test_days: int,
    embargo_days: int = 0,
    step_days: int = None,
) -> Iterator[WalkForwardFold]:
    """
    Generate walk-forward train/test date splits.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        All available dates (will be sorted and deduplicated).
    train_days : int
        Calendar days for training window.
    test_days : int
        Calendar days for testing window.
    embargo_days : int
        Gap between train end and test start. Set to at least your
        forward-return horizon to prevent leakage.
    step_days : int, optional
        Step size between folds. Defaults to test_days (non-overlapping tests).

    Yields
    ------
    WalkForwardFold
    """
    if step_days is None:
        step_days = test_days

    dates = pd.DatetimeIndex(dates).sort_values().unique()
    if len(dates) == 0:
        return

    start_date = dates[0]
    end_date = dates[-1]
    total_span = (end_date - start_date).days

    min_required = train_days + embargo_days + test_days
    if total_span < min_required:
        raise ValueError(
            f"Date range spans {total_span} days, but need at least "
            f"{min_required} (train={train_days} + embargo={embargo_days} "
            f"+ test={test_days})."
        )

    fold_idx = 0
    cursor = start_date

    while True:
        train_start = cursor
        train_end = cursor + pd.Timedelta(days=train_days - 1)
        test_start = train_end + pd.Timedelta(days=embargo_days + 1)
        test_end = test_start + pd.Timedelta(days=test_days - 1)

        if test_end > end_date:
            break

        train_dates = dates[(dates >= train_start) & (dates <= train_end)]
        test_dates = dates[(dates >= test_start) & (dates <= test_end)]

        if len(train_dates) < 10 or len(test_dates) < 5:
            cursor += pd.Timedelta(days=step_days)
            continue

        yield WalkForwardFold(
            fold_idx=fold_idx,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
        )

        fold_idx += 1
        cursor += pd.Timedelta(days=step_days)


def select_dates(
    data: pd.DataFrame,
    fold: WalkForwardFold,
    split: str = "train",
) -> pd.DataFrame:
    """
    Select rows from data belonging to a fold's train or test period.

    Parameters
    ----------
    data : pd.DataFrame
        Must have a DatetimeIndex.
    fold : WalkForwardFold
    split : 'train' or 'test'
    """
    if split == "train":
        mask = (data.index >= fold.train_start) & (data.index <= fold.train_end)
    elif split == "test":
        mask = (data.index >= fold.test_start) & (data.index <= fold.test_end)
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return data.loc[mask]
