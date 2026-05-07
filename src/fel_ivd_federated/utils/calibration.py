"""Calibration metrics: ECE, Brier score, reliability diagram data."""

import numpy as np


def expected_calibration_error(y_true, proba, classes, n_bins=15):
    """Compute Expected Calibration Error (ECE).

    For each sample the confidence is max(predicted proba) and the
    correctness is 1 if argmax matches y_true.  Samples are binned by
    confidence; ECE = sum_b (|bin| / N) * |acc_b - conf_b|.

    Parameters
    ----------
    y_true : array-like of str/int  (N,)
    proba  : ndarray (N, K) predicted probabilities
    classes: list of class labels (length K), same order as proba columns
    n_bins : int

    Returns
    -------
    ece : float
    """
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_idx[v] for v in y_true])
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correct = (predictions == y_idx).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if lo == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correct[mask].mean()
        ece += (count / n) * abs(avg_acc - avg_conf)
    return float(ece)


def brier_score(y_true, proba, classes):
    """Multiclass Brier score = mean over samples of ||p - one_hot(y)||^2.

    Parameters
    ----------
    y_true : array-like of str/int  (N,)
    proba  : ndarray (N, K)
    classes: list of class labels

    Returns
    -------
    score : float   (lower is better; 0 = perfect)
    """
    cls_idx = {c: i for i, c in enumerate(classes)}
    n = len(y_true)
    K = len(classes)
    one_hot = np.zeros((n, K), dtype=np.float64)
    for i, v in enumerate(y_true):
        one_hot[i, cls_idx[v]] = 1.0
    return float(((proba - one_hot) ** 2).sum(axis=1).mean())


def reliability_diagram_data(y_true, proba, classes, n_bins=15):
    """Return per-bin (mean_confidence, mean_accuracy, bin_count) for
    plotting a reliability diagram.

    Returns
    -------
    bins : list of dicts  [{conf, acc, count, lo, hi}, ...]
    """
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_idx[v] for v in y_true])
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correct = (predictions == y_idx).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if lo == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"conf": float((lo + hi) / 2), "acc": 0.0, "count": 0,
                         "lo": float(lo), "hi": float(hi)})
        else:
            bins.append({"conf": float(confidences[mask].mean()),
                         "acc": float(correct[mask].mean()),
                         "count": count,
                         "lo": float(lo), "hi": float(hi)})
    return bins
