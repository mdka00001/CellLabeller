"""scVI-based integration of reference and query AnnData objects.

Creates a shared latent space using scVI (``scvi-tools``) trained on the
reference dataset.  The query dataset is then projected into the same
latent space using scVI's built-in *out-of-sample* mapping so that
reference and query cells are directly comparable.

Usage
-----
>>> from celllabeller.integration import integrate_datasets
>>> ref_latent, query_latent = integrate_datasets(ref_adata, query_adata)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import anndata as ad
import numpy as np
import scvi

logger = logging.getLogger(__name__)


def _get_common_genes(ref: ad.AnnData, query: ad.AnnData) -> list[str]:
    """Return sorted list of genes present in both *ref* and *query*."""
    common = sorted(set(ref.var_names) & set(query.var_names))
    if not common:
        raise ValueError(
            "Reference and query datasets share no common genes. "
            "Ensure var_names represent the same gene identifier space."
        )
    logger.info("Found %d common genes between reference and query.", len(common))
    return common


def _subset_to_common(
    ref: ad.AnnData, query: ad.AnnData
) -> Tuple[ad.AnnData, ad.AnnData, list[str]]:
    """Subset both datasets to common genes (in the same order)."""
    common = _get_common_genes(ref, query)
    return ref[:, common].copy(), query[:, common].copy(), common


def integrate_datasets(
    ref_adata: ad.AnnData,
    query_adata: ad.AnnData,
    *,
    n_latent: int = 30,
    n_layers: int = 2,
    n_hidden: int = 128,
    max_epochs: int = 400,
    early_stopping: bool = True,
    batch_key: Optional[str] = None,
    train_size: float = 0.9,
    seed: int = 42,
    accelerator: str = "cpu",
) -> Tuple[ad.AnnData, ad.AnnData, scvi.model.SCVI]:
    """Integrate *ref_adata* and *query_adata* with scVI.

    Parameters
    ----------
    ref_adata:
        Reference AnnData (cells × genes, raw counts in ``X``).
    query_adata:
        Query AnnData (cells × genes, raw counts in ``X``).
    n_latent:
        Dimensionality of the scVI latent space.
    n_layers:
        Number of hidden layers in the encoder/decoder.
    n_hidden:
        Number of units per hidden layer.
    max_epochs:
        Maximum training epochs.
    early_stopping:
        Whether to enable early stopping.
    batch_key:
        ``obs`` column to use as batch covariate (``None`` = no correction).
    train_size:
        Fraction of reference cells used for training.
    seed:
        Random seed for reproducibility.
    accelerator:
        PyTorch-Lightning accelerator (``"cpu"``, ``"gpu"``, …).

    Returns
    -------
    ref_adata_sub : ad.AnnData
        Reference subset to common genes; scVI latent stored in
        ``obsm["X_scVI"]``.
    query_adata_sub : ad.AnnData
        Query subset to common genes; scVI latent stored in
        ``obsm["X_scVI"]``.
    model : scvi.model.SCVI
        Trained scVI model (reference).
    """
    scvi.settings.seed = seed

    ref_sub, query_sub, common_genes = _subset_to_common(ref_adata, query_adata)
    logger.info(
        "Reference: %d cells × %d genes | Query: %d cells × %d genes",
        ref_sub.n_obs,
        ref_sub.n_vars,
        query_sub.n_obs,
        query_sub.n_vars,
    )

    # ---- train on reference --------------------------------------------
    scvi.model.SCVI.setup_anndata(ref_sub, batch_key=batch_key)
    model = scvi.model.SCVI(
        ref_sub,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
    )
    model.train(
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        train_size=train_size,
        accelerator=accelerator,
    )

    ref_sub.obsm["X_scVI"] = model.get_latent_representation()
    logger.info("Reference latent representation stored in obsm['X_scVI'].")

    # ---- map query into the reference latent space ---------------------
    scvi.model.SCVI.prepare_query_anndata(query_sub, model)
    query_model = scvi.model.SCVI.load_query_data(query_sub, model)
    query_model.train(
        max_epochs=200,
        plan_kwargs={"weight_decay": 0.0},
        early_stopping=True,
        accelerator=accelerator,
    )
    query_sub.obsm["X_scVI"] = query_model.get_latent_representation()
    logger.info("Query latent representation stored in obsm['X_scVI'].")

    return ref_sub, query_sub, model
