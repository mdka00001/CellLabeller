"""Tests for the CellLabeller pipeline (integration mocked)."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import anndata as ad
import pandas as pd


def _make_integrated_pair(ref_adata, query_adata):
    """Simulate what integrate_datasets() returns (mocked latent)."""
    rng = np.random.default_rng(99)
    ref = ref_adata.copy()
    query = query_adata.copy()
    ref.obsm["X_scVI"] = rng.standard_normal((ref.n_obs, 10)).astype(np.float32)
    query.obsm["X_scVI"] = rng.standard_normal((query.n_obs, 10)).astype(np.float32)
    mock_model = MagicMock()
    mock_model.get_latent_representation.return_value = query.obsm["X_scVI"]
    return ref, query, mock_model


class TestCellLabellerFit:
    def test_fit_stores_components(self, ref_adata, query_adata):
        from celllabeller import CellLabeller

        cl = CellLabeller(
            celltype_key="cell_type",
            hyperparameter_search=False,
            feature_mode="combined",
        )

        with patch("celllabeller.pipeline.integrate_datasets") as mock_int:
            ref_sub, query_sub, mock_model = _make_integrated_pair(ref_adata, query_adata)
            mock_int.return_value = (ref_sub, query_sub, mock_model)
            cl.fit(ref_adata, query_adata)

        assert cl.clf_ is not None
        assert cl.label_encoder_ is not None
        assert cl.feature_names_ is not None
        assert cl.common_genes_ is not None

    def test_fit_wrong_key_raises(self, ref_adata, query_adata):
        from celllabeller import CellLabeller

        cl = CellLabeller(celltype_key="nonexistent_key", hyperparameter_search=False)
        with pytest.raises(ValueError, match="nonexistent_key"):
            cl.fit(ref_adata, query_adata)


class TestCellLabellerLabelQuery:
    def test_label_query_populates_obs(self, ref_adata, query_adata):
        from celllabeller import CellLabeller

        cl = CellLabeller(
            celltype_key="cell_type",
            hyperparameter_search=False,
            feature_mode="combined",
        )
        with patch("celllabeller.pipeline.integrate_datasets") as mock_int:
            ref_sub, query_sub, mock_model = _make_integrated_pair(ref_adata, query_adata)
            mock_int.return_value = (ref_sub, query_sub, mock_model)
            cl.fit(ref_adata, query_adata)

        annotated = cl.label_query()
        assert "predicted_celltype" in annotated.obs.columns
        assert "prediction_confidence" in annotated.obs.columns
        assert "prediction_proba" in annotated.obsm
        assert annotated.n_obs == query_adata.n_obs

    def test_not_fitted_raises(self, ref_adata):
        from celllabeller import CellLabeller

        cl = CellLabeller()
        with pytest.raises(RuntimeError, match="not fitted"):
            cl.label_query(ref_adata)


class TestCellLabellerScore:
    def test_score_returns_metrics(self, ref_adata, query_adata):
        from celllabeller import CellLabeller
        import pandas as pd

        cl = CellLabeller(
            celltype_key="cell_type",
            hyperparameter_search=False,
            feature_mode="combined",
        )
        with patch("celllabeller.pipeline.integrate_datasets") as mock_int:
            ref_sub, query_sub, mock_model = _make_integrated_pair(ref_adata, query_adata)
            # Give query ground-truth labels too
            query_sub.obs["cell_type"] = np.random.default_rng(5).choice(
                ["T cell", "B cell", "NK cell"], query_sub.n_obs
            )
            mock_int.return_value = (ref_sub, query_sub, mock_model)
            cl.fit(ref_adata, query_adata)

        cl.label_query()
        metrics = cl.score(cl.query_adata_)
        assert "accuracy" in metrics
        assert "f1" in metrics


class TestCellLabellerSaveLoad:
    def test_save_and_load(self, tmp_path, ref_adata, query_adata):
        from celllabeller import CellLabeller

        cl = CellLabeller(
            celltype_key="cell_type",
            hyperparameter_search=False,
            feature_mode="combined",
        )
        with patch("celllabeller.pipeline.integrate_datasets") as mock_int:
            ref_sub, query_sub, mock_model = _make_integrated_pair(ref_adata, query_adata)
            mock_int.return_value = (ref_sub, query_sub, mock_model)
            cl.fit(ref_adata, query_adata)

        bundle_dir = cl.save(tmp_path / "bundle")

        cl2 = CellLabeller.load(bundle_dir)
        assert cl2.clf_ is not None
        assert cl2.label_encoder_ is not None
        assert set(cl2.label_encoder_.classes_) == set(cl.label_encoder_.classes_)


class TestCellLabellerRepr:
    def test_repr(self, ref_adata, query_adata):
        from celllabeller import CellLabeller

        cl = CellLabeller(feature_mode="latent", n_latent=20)
        assert "CellLabeller" in repr(cl)
        assert "fitted=False" in repr(cl)
