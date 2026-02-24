"""Scoring and visualisation utilities for CellLabeller predictions.

Functions
---------
score_predictions  – compute classification metrics.
plot_confusion_matrix  – annotated heatmap of the confusion matrix.
plot_umap  – UMAP embedding coloured by cell-type labels.
plot_prediction_confidence  – violin / box plot of per-cell confidence scores.
plot_feature_importance  – bar chart of top-N XGBoost feature importances.

Usage
-----
>>> from celllabeller.evaluation import score_predictions, plot_confusion_matrix
>>> metrics = score_predictions(y_true, y_pred, y_proba, label_encoder=le)
>>> fig = plot_confusion_matrix(y_true, y_pred, labels=le.classes_)
>>> fig.savefig("confusion_matrix.png")
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports – warn gracefully if plotting libraries are absent
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _PLOT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLOT_AVAILABLE = False
    logger.warning(
        "matplotlib / seaborn not installed; plotting functions will fail."
    )

try:
    import scanpy as sc

    _SCANPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCANPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    *,
    label_encoder=None,
    average: str = "weighted",
) -> dict:
    """Compute classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels (same type as *y_true*).
    y_proba:
        Predicted probabilities, shape ``(n_cells, n_classes)``.
        Required for ``roc_auc`` (multi-class OvR).
    label_encoder:
        Fitted :class:`~sklearn.preprocessing.LabelEncoder`; used to
        decode integer labels before computing per-class metrics.
    average:
        Averaging strategy for precision/recall/F1
        (``"weighted"``, ``"macro"``, ``"micro"``).

    Returns
    -------
    metrics : dict
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1``, and optionally ``roc_auc``.
    """
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    metrics: dict = {}

    # Decode to strings if label_encoder is provided
    if label_encoder is not None:
        def _to_int(arr):
            is_str = arr.dtype.kind in ("U", "S", "O")
            return label_encoder.transform(arr) if is_str else arr

        y_true_dec = label_encoder.inverse_transform(_to_int(y_true))
        y_pred_dec = label_encoder.inverse_transform(_to_int(y_pred))
    else:
        y_true_dec, y_pred_dec = y_true, y_pred

    metrics["accuracy"] = float(accuracy_score(y_true_dec, y_pred_dec))
    metrics["precision"] = float(
        precision_score(y_true_dec, y_pred_dec, average=average, zero_division=0)
    )
    metrics["recall"] = float(
        recall_score(y_true_dec, y_pred_dec, average=average, zero_division=0)
    )
    metrics["f1"] = float(
        f1_score(y_true_dec, y_pred_dec, average=average, zero_division=0)
    )

    if y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score  # noqa: PLC0415

            n_classes = y_proba.shape[1]
            roc_multi = "ovr" if n_classes > 2 else "raise"
            metrics["roc_auc"] = float(
                roc_auc_score(
                    y_true_dec,
                    y_proba,
                    multi_class=roc_multi,
                    average=average,
                    labels=label_encoder.classes_ if label_encoder else None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not compute ROC-AUC: %s", exc)

    metrics["classification_report"] = classification_report(
        y_true_dec, y_pred_dec, zero_division=0
    )
    logger.info("Accuracy: %.4f  F1 (%s): %.4f", metrics["accuracy"], average, metrics["f1"])
    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _require_plot():
    if not _PLOT_AVAILABLE:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install them with: pip install matplotlib seaborn"
        )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Optional[Sequence[str]] = None,
    normalize: bool = True,
    figsize: tuple = (10, 8),
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
) -> "plt.Figure":
    """Plot an annotated confusion matrix heatmap.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    labels:
        Ordered list of class labels for axis ticks.
    normalize:
        If ``True``, normalise counts to row fractions.
    figsize:
        Figure size ``(width, height)`` in inches.
    cmap:
        Matplotlib colour map name.
    title:
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _require_plot()
    from sklearn.metrics import confusion_matrix  # noqa: PLC0415

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=figsize)
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_umap(
    adata,
    *,
    color_key: str = "predicted_celltype",
    title: str = "UMAP – predicted cell types",
    figsize: tuple = (8, 6),
    use_scanpy: bool = True,
) -> "plt.Figure":
    """Plot a UMAP embedding coloured by cell-type labels.

    Requires ``obsm["X_umap"]`` to be pre-computed (or ``use_scanpy=True``
    to compute it on-the-fly from ``obsm["X_scVI"]``).

    Parameters
    ----------
    adata:
        AnnData with ``obsm["X_umap"]`` (or ``obsm["X_scVI"]`` if
        ``use_scanpy=True``).
    color_key:
        ``obs`` column to colour cells by.
    title:
        Plot title.
    figsize:
        Figure size.
    use_scanpy:
        If ``True`` and ``X_umap`` is absent, compute UMAP via scanpy
        using the scVI latent space as the neighbour graph input.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _require_plot()
    if not _SCANPY_AVAILABLE:
        raise ImportError("scanpy is required for plot_umap. pip install scanpy")

    if "X_umap" not in adata.obsm and use_scanpy:
        logger.info("Computing UMAP from scVI latent space …")
        sc.pp.neighbors(adata, use_rep="X_scVI")
        sc.tl.umap(adata)

    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.umap(adata, color=color_key, ax=ax, show=False, title=title)
    plt.tight_layout()
    return fig


def plot_prediction_confidence(
    adata,
    *,
    confidence_key: str = "prediction_confidence",
    groupby: str = "predicted_celltype",
    figsize: tuple = (12, 5),
    title: str = "Prediction Confidence per Cell Type",
) -> "plt.Figure":
    """Violin plot of per-cell prediction confidence grouped by cell type.

    Parameters
    ----------
    adata:
        AnnData with ``obs[confidence_key]`` and ``obs[groupby]``.
    confidence_key:
        Column in ``obs`` holding the maximum predicted probability.
    groupby:
        Column in ``obs`` defining the groups (cell types).
    figsize:
        Figure size.
    title:
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _require_plot()
    if confidence_key not in adata.obs.columns:
        raise KeyError(
            f"'{confidence_key}' not found in adata.obs. "
            "Run CellLabeller.label_query() first."
        )

    df = adata.obs[[groupby, confidence_key]].copy()
    order = (
        df.groupby(groupby)[confidence_key].median().sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=df,
        x=groupby,
        y=confidence_key,
        order=order,
        inner="quartile",
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Predicted cell type")
    ax.set_ylabel("Confidence (max probability)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    clf,
    feature_names: Optional[Sequence[str]] = None,
    *,
    top_n: int = 30,
    importance_type: str = "gain",
    figsize: tuple = (10, 8),
    title: str = "XGBoost Feature Importance",
) -> "plt.Figure":
    """Bar chart of the top-N XGBoost feature importances.

    Parameters
    ----------
    clf:
        Trained :class:`~xgboost.XGBClassifier`.
    feature_names:
        List of feature names (length must equal ``clf.n_features_in_``).
    top_n:
        Show only the *top_n* most important features.
    importance_type:
        XGBoost importance type: ``"weight"``, ``"gain"``, or ``"cover"``.
    figsize:
        Figure size.
    title:
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _require_plot()
    scores = clf.get_booster().get_score(importance_type=importance_type)

    if feature_names is not None:
        # Remap ``f0``, ``f1``, … to human-readable names
        renamed = {}
        for k, v in scores.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if idx < len(feature_names):
                    renamed[feature_names[idx]] = v
                else:
                    renamed[k] = v
            else:
                renamed[k] = v
        scores = renamed

    series = pd.Series(scores).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    series[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel(f"Importance ({importance_type})")
    ax.set_title(title)
    plt.tight_layout()
    return fig
