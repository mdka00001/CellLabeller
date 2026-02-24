"""Tests for celllabeller.features."""
import numpy as np
import pytest

from celllabeller.features import build_feature_matrix, feature_dataframe


class TestGeneFeatures:
    def test_shape(self, ref_adata):
        X, names = build_feature_matrix(ref_adata, mode="gene")
        assert X.shape == (ref_adata.n_obs, ref_adata.n_vars)
        assert len(names) == ref_adata.n_vars

    def test_feature_name_prefix(self, ref_adata):
        _, names = build_feature_matrix(ref_adata, mode="gene")
        assert all(n.startswith("gene:") for n in names)

    def test_nonnegative(self, ref_adata):
        X, _ = build_feature_matrix(ref_adata, mode="gene")
        assert np.all(X >= 0)


class TestLatentFeatures:
    def test_shape(self, ref_with_latent):
        X, names = build_feature_matrix(ref_with_latent, mode="latent")
        assert X.shape[0] == ref_with_latent.n_obs
        assert X.shape[1] == ref_with_latent.obsm["X_scVI"].shape[1]

    def test_feature_name_prefix(self, ref_with_latent):
        _, names = build_feature_matrix(ref_with_latent, mode="latent")
        assert all(n.startswith("scVI:") for n in names)

    def test_missing_latent_raises(self, ref_adata):
        with pytest.raises(KeyError, match="X_scVI"):
            build_feature_matrix(ref_adata, mode="latent")


class TestCombinedFeatures:
    def test_shape(self, ref_with_latent):
        X, names = build_feature_matrix(ref_with_latent, mode="combined")
        n_genes = ref_with_latent.n_vars
        n_latent = ref_with_latent.obsm["X_scVI"].shape[1]
        assert X.shape == (ref_with_latent.n_obs, n_genes + n_latent)
        assert len(names) == n_genes + n_latent

    def test_both_prefixes(self, ref_with_latent):
        _, names = build_feature_matrix(ref_with_latent, mode="combined")
        assert any(n.startswith("gene:") for n in names)
        assert any(n.startswith("scVI:") for n in names)


class TestUnknownMode:
    def test_unknown_mode_raises(self, ref_with_latent):
        with pytest.raises(ValueError, match="Unknown mode"):
            build_feature_matrix(ref_with_latent, mode="invalid")


class TestFeatureDataframe:
    def test_returns_dataframe(self, ref_with_latent):
        import pandas as pd

        df = feature_dataframe(ref_with_latent, mode="combined")
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == ref_with_latent.n_obs
