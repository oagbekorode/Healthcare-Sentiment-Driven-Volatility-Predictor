import pandas as pd

from pipeline.experiment import safe_auc, walk_forward_windows


def test_walk_forward_windows_has_expanding_splits():
    splits = walk_forward_windows(n=180, min_train=80, n_splits=3)
    assert len(splits) >= 2
    assert splits[0][0] == 80
    assert splits[0][1] > splits[0][0]
    assert splits[-1][0] > splits[0][0]


def test_safe_auc_perfect_ranking_is_one():
    y = pd.Series([0, 0, 1, 1])
    score = pd.Series([0.1, 0.2, 0.8, 0.9])
    assert safe_auc(y, score) == 1.0

