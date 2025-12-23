from __future__ import annotations

import numpy as np


def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC for binary labels without sklearn.

    Args:
        y_true: shape (n,), values in {0,1}
        y_score: shape (n,), higher means more likely positive

    Returns:
        AUC in [0,1]. If the metric is undefined (all positives or all negatives), returns NaN.
    """

    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)

    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank scores; handle ties by average rank.
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # Average ranks for ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1

    sum_pos_ranks = float(ranks[pos].sum())
    # Mann–Whitney U statistic
    u = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    auc = u / (n_pos * n_neg)
    return float(auc)
