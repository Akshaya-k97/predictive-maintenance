"""Metric and curve helpers. Every number shown in the dashboard comes from here."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Metrics for a given probability threshold (failure = class 1)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _downsample(arrays, max_points):
    n = len(arrays[0])
    if n <= max_points:
        return [a.tolist() for a in arrays]
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return [a[idx].tolist() for a in arrays]


def curve_data(y_true, y_proba, max_points: int = 200) -> dict:
    """ROC and precision-recall curve points (downsampled to keep JSON small)."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    fpr, tpr = _downsample([fpr, tpr], max_points)
    prec, rec = _downsample([prec, rec], max_points)
    return {"roc": {"fpr": fpr, "tpr": tpr}, "pr": {"precision": prec, "recall": rec}}


def best_threshold(y_true, y_proba, beta: float = 2.0):
    """Threshold maximising F-beta (beta=2 weights recall higher than precision).

    Must only be called with TRAINING-side predictions (e.g. out-of-fold),
    never with test-set predictions.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    prec, rec = prec[:-1], rec[:-1]
    denom = beta ** 2 * prec + rec
    safe = np.where(denom > 0, denom, 1.0)
    f = np.where(denom > 0, (1 + beta ** 2) * prec * rec / safe, 0.0)
    i = int(np.argmax(f))
    return float(np.clip(thr[i], 0.05, 0.95)), float(f[i])
