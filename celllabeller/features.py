"""Feature engineering for the CellLabeller classifier.

Generates two complementary feature matrices from an AnnData object:

* **Gene features** – log1p-normalised expression of the common gene set.
* **Latent features** – scVI latent coordinates (``obsm["X_scVI"]``).
* **Combined features** – horizontal concatenation of both.

Usage
-----
>>> from celllabeller.features import build_feature_matrix
>>> X, feature_names = build_feature_matrix(adata, mode="combined")
"""

from __future__ import annotations

import logging
from typing import Literal, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

FeatureMode = Literal["gene", "latent", "combined"]


def _dense(x: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Convert sparse or dense matrix to a dense float32 ndarray."""
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _gene_features(
    adata: ad.AnnData,
    *,
    target_sum: float = 1e4,
    log1p: bool = True,
) -> Tuple[np.ndarray, list[str]]:
    """Return normalised gene expression matrix and gene feature names.

    Parameters
    ----------
    adata:
        AnnData with raw counts in ``X``.
    target_sum:
        Total counts per cell to normalise to (``None`` = skip normalisation).
    log1p:
        Whether to apply log1p transformation after normalisation.
    """
    X = _dense(adata.X)

    if target_sum is not None:
        cell_totals = X.sum(axis=1, keepdims=True)
        # avoid division by zero for empty cells
        cell_totals = np.where(cell_totals == 0, 1.0, cell_totals)
        X = X / cell_totals * target_sum

    if log1p:
        np.log1p(X, out=X)

    feature_names = [f"gene:{g}" for g in adata.var_names.tolist()]
    logger.debug("Gene features: %d cells × %d genes.", X.shape[0], X.shape[1])
    return X, feature_names


def _latent_features(adata: ad.AnnData) -> Tuple[np.ndarray, list[str]]:
    """Return scVI latent coordinates stored in ``obsm["X_scVI"]``."""
    if "X_scVI" not in adata.obsm:
        raise KeyError(
            "'X_scVI' not found in adata.obsm. "
            "Run integrate_datasets() first to populate the latent space."
        )
    Z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    feature_names = [f"scVI:{i}" for i in range(Z.shape[1])]
    logger.debug("Latent features: %d cells × %d dims.", Z.shape[0], Z.shape[1])
    return Z, feature_names


def build_feature_matrix(
    adata: ad.AnnData,
    mode: FeatureMode = "combined",
    *,
    target_sum: float = 1e4,
    log1p: bool = True,
) -> Tuple[np.ndarray, list[str]]:
    """Build a feature matrix from *adata*.

    Parameters
    ----------
    adata:
        AnnData after integration (must contain ``obsm["X_scVI"]`` for
        ``mode="latent"`` or ``mode="combined"``).
    mode:
        Which features to include:

        * ``"gene"``     – log1p-normalised expression only.
        * ``"latent"``   – scVI latent coordinates only.
        * ``"combined"`` – both concatenated horizontally (default).
    target_sum:
        Normalisation target per cell (applies to gene features only).
    log1p:
        Apply log1p to gene expression (applies to gene features only).

    Returns
    -------
    X : np.ndarray, shape (n_cells, n_features)
        Feature matrix.
    feature_names : list[str]
        Name of each feature column (useful for SHAP / feature importance).
    """
    if mode == "gene":
        return _gene_features(adata, target_sum=target_sum, log1p=log1p)

    if mode == "latent":
        return _latent_features(adata)

    if mode == "combined":
        X_gene, names_gene = _gene_features(adata, target_sum=target_sum, log1p=log1p)
        X_lat, names_lat = _latent_features(adata)
        X = np.concatenate([X_gene, X_lat], axis=1)
        logger.info(
            "Combined features: %d gene + %d latent = %d total.",
            X_gene.shape[1],
            X_lat.shape[1],
            X.shape[1],
        )
        return X, names_gene + names_lat

    raise ValueError(f"Unknown mode '{mode}'. Choose 'gene', 'latent', or 'combined'.")


def feature_dataframe(
    adata: ad.AnnData,
    mode: FeatureMode = "combined",
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper returning a labelled :class:`pd.DataFrame`."""
    X, names = build_feature_matrix(adata, mode, **kwargs)
    return pd.DataFrame(X, index=adata.obs_names, columns=names)
