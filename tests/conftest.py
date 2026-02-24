"""Shared fixtures for CellLabeller tests.

All fixtures use tiny synthetic datasets so tests run fast without
real scVI training.  scVI integration is mocked by directly writing a
random latent matrix into ``obsm["X_scVI"]``.
"""
from __future__ import annotations

import numpy as np
import pytest
import anndata as ad
import pandas as pd
import scipy.sparse as sp

# Negative-binomial parameters for synthetic count data
_NB_N = 5
_NB_P = 0.5


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_adata(n_cells: int, n_genes: int, *, celltypes=None, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(_NB_N, _NB_P, size=(n_cells, n_genes)).astype(np.float32)
    gene_names = [f"gene{i}" for i in range(n_genes)]
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    if celltypes is not None:
        obs["cell_type"] = np.random.default_rng(seed).choice(celltypes, n_cells)
    var = pd.DataFrame(index=gene_names)
    return ad.AnnData(X=X, obs=obs, var=var)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def ref_adata():
    """Reference AnnData with 200 cells, 50 genes, 3 cell types."""
    return _make_adata(200, 50, celltypes=["T cell", "B cell", "NK cell"], seed=1)


@pytest.fixture(scope="session")
def query_adata():
    """Query AnnData with 100 cells and the same 50 genes (no labels)."""
    return _make_adata(100, 50, seed=2)


@pytest.fixture(scope="session")
def ref_with_latent(ref_adata):
    """Reference AnnData with a fake scVI latent matrix injected."""
    adata = ref_adata.copy()
    rng = np.random.default_rng(10)
    adata.obsm["X_scVI"] = rng.standard_normal((adata.n_obs, 10)).astype(np.float32)
    return adata


@pytest.fixture(scope="session")
def query_with_latent(query_adata):
    """Query AnnData with a fake scVI latent matrix injected."""
    adata = query_adata.copy()
    rng = np.random.default_rng(11)
    adata.obsm["X_scVI"] = rng.standard_normal((adata.n_obs, 10)).astype(np.float32)
    return adata


@pytest.fixture(scope="session")
def trained_clf(ref_with_latent):
    """A tiny trained XGBClassifier + LabelEncoder for reuse."""
    from celllabeller.features import build_feature_matrix
    from celllabeller.classifier import train_classifier

    X, _ = build_feature_matrix(ref_with_latent, mode="combined")
    y = ref_with_latent.obs["cell_type"].values
    clf, le = train_classifier(X, y, params={"n_estimators": 10, "max_depth": 3})
    return clf, le
