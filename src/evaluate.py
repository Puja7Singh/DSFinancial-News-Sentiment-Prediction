"""Metrics, confusion matrices and error-analysis helpers shared by all models."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score)

from . import config  # noqa: E402


def compute_metrics(y_true, y_pred):
    """Accuracy, macro F1 and class-wise precision/recall/F1 as a dict."""
    report = classification_report(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)),
        target_names=config.LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": {name: {k: report[name][k]
                             for k in ("precision", "recall", "f1-score",
                                       "support")}
                      for name in config.LABELS},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(config.NUM_CLASSES))).tolist(),
    }


def save_metrics(name, metrics, extra=None):
    """Write ``reports/metrics/<name>.json``."""
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": name, **metrics, **(extra or {})}
    path = config.METRICS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_all_metrics():
    """Load every metrics JSON into {model_name: metrics}."""
    out = {}
    for path in sorted(config.METRICS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        out[data["model"]] = data
    return out


def plot_confusion(cm, title, path=None, ax=None):
    """Row-normalised confusion matrix with raw counts annotated."""
    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(config.LABELS)), config.LABELS)
    ax.set_yticks(range(len(config.LABELS)), config.LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center",
                    va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "black")
    if own_fig:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
        plt.close(fig)


def misclassified(texts, y_true, y_pred, probs=None, n=15):
    """Return the most confident mistakes as a list of dicts."""
    rows = []
    for i, (t, a, b) in enumerate(zip(texts, y_true, y_pred)):
        if a != b:
            conf = float(probs[i][b]) if probs is not None else None
            rows.append({"text": t, "true": config.ID2LABEL[int(a)],
                         "pred": config.ID2LABEL[int(b)], "confidence": conf})
    if probs is not None:
        rows.sort(key=lambda r: -r["confidence"])
    return rows[:n]
