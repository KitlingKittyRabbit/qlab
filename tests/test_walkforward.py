import pandas as pd
import pytest

from qlab.walkforward import walk_forward_splits, select_dates


class TestWalkForwardSplits:

    def _dates(self, n=500):
        return pd.date_range("2020-01-01", periods=n, freq="D")

    def test_produces_folds(self):
        folds = list(walk_forward_splits(self._dates(), train_days=90, test_days=90))
        assert len(folds) >= 2

    def test_train_before_test(self):
        for fold in walk_forward_splits(self._dates(), train_days=90, test_days=90):
            assert fold.train_end < fold.test_start

    def test_embargo_gap(self):
        folds = list(walk_forward_splits(
            self._dates(), train_days=90, test_days=90, embargo_days=14
        ))
        for fold in folds:
            gap = (fold.test_start - fold.train_end).days
            assert gap >= 14

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="need at least"):
            list(walk_forward_splits(self._dates(100), train_days=90, test_days=90))

    def test_fold_indices_sequential(self):
        folds = list(walk_forward_splits(self._dates(), train_days=90, test_days=90))
        for i, fold in enumerate(folds):
            assert fold.fold_idx == i

    def test_non_overlapping_tests(self):
        folds = list(walk_forward_splits(self._dates(), train_days=90, test_days=90))
        for i in range(1, len(folds)):
            assert folds[i].test_start > folds[i - 1].test_end


class TestSelectDates:

    def test_correct_split(self):
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        data = pd.DataFrame({"x": range(500)}, index=dates)
        folds = list(walk_forward_splits(dates, train_days=90, test_days=90))
        fold = folds[0]

        train = select_dates(data, fold, "train")
        test = select_dates(data, fold, "test")

        assert len(train) > 0
        assert len(test) > 0
        assert train.index.max() < test.index.min()

    def test_invalid_split_raises(self):
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        data = pd.DataFrame({"x": range(500)}, index=dates)
        folds = list(walk_forward_splits(dates, train_days=90, test_days=90))
        with pytest.raises(ValueError, match="split"):
            select_dates(data, folds[0], "invalid")
