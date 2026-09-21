"""
Evaluation metrics for imbalanced behavioural anomaly detection.

Why accuracy is not reported as a primary metric
------------------------------------------------
Under our injection rates the anomaly class is a small minority of the
evaluation window. A detector that flags nothing attains an accuracy equal to
the genuine-class prevalence while detecting nothing at all. Accuracy therefore
cannot distinguish a useful detector from an inert one in this regime, and any
change in prevalence changes accuracy without any change in the detector. We
report PR-AUC as the primary ranking metric because precision and recall both
condition on the minority class, and we report a prevalence-independent
ranking metric (ROC-AUC) alongside it.

Threshold selection
-------------------
Operating-point metrics require a threshold. Selecting a threshold on the test
labels would leak. We instead set the threshold from the training history: flag
a transaction when its score exceeds the (1 - alpha) quantile of the scores the
fitted model assigns to its own training data. This is the rule a deployed
system could actually apply, and it degrades in a visible way when the history
is too short to estimate a quantile, which is part of what we are measuring.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision. NaN when only one class is present."""
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(average_precision_score(labels, scores))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def lift_over_random(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    PR-AUC divided by the prevalence baseline. A random ranker scores 1.0.
    This makes PR-AUC comparable across evaluation windows whose anomaly
    prevalence differs slightly because of integer rounding.
    """
    p = float(np.mean(labels))
    if p <= 0 or p >= 1:
        return np.nan
    ap = pr_auc(labels, scores)
    return ap / p if not np.isnan(ap) else np.nan


def training_quantile_threshold(
    train_scores: np.ndarray, alpha: float = 0.05
) -> float:
    """
    Threshold at the (1 - alpha) quantile of training scores.

    With a single training point the quantile equals that point's score and the
    rule becomes a near-degenerate "anything above my one observation". We
    return it anyway and let the metrics show the consequence.
    """
    if len(train_scores) == 0:
        return np.inf
    return float(np.quantile(train_scores, 1.0 - alpha))


def operating_point_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    """Precision, recall, F1, FPR and FNR at a fixed threshold."""
    pred = (scores > threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if precision is np.nan or recall is np.nan or (precision + recall) in (0, np.nan):
        f1 = 0.0 if (tp + fp + fn) > 0 else np.nan
    else:
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        "flag_rate": float(pred.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def detection_latency(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """
    Index of the first correctly flagged anomaly within the evaluation window,
    expressed as a count of transactions from the window start. NaN when no
    anomaly is ever flagged. The window is chronologically ordered, so this is
    a transaction-count latency rather than a wall-clock latency.
    """
    flagged = (scores > threshold) & (labels == 1)
    idx = np.flatnonzero(flagged)
    return float(idx[0]) if len(idx) else np.nan


def evaluate(
    labels: np.ndarray,
    test_scores: np.ndarray,
    train_scores: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Full metric bundle for one account, one detector, one history size."""
    thr = training_quantile_threshold(train_scores, alpha=alpha)
    out = {
        "pr_auc": pr_auc(labels, test_scores),
        "roc_auc": roc_auc(labels, test_scores),
        "lift": lift_over_random(labels, test_scores),
        "threshold": thr,
        "latency": detection_latency(labels, test_scores, thr),
    }
    out.update(operating_point_metrics(labels, test_scores, thr))
    return out
