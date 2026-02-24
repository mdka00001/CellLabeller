"""Tests for celllabeller.models (save/load)."""
import pathlib

import pytest
from sklearn.preprocessing import LabelEncoder

from celllabeller.features import build_feature_matrix
from celllabeller.classifier import train_classifier
from celllabeller.models import save_model_bundle, load_model_bundle, save_pipeline, load_pipeline


class _DummyPipeline:
    """Module-level dummy class so joblib can pickle it."""
    value = 42


class TestModelBundle:
    def test_save_and_load(self, tmp_path, ref_with_latent):
        X, _ = build_feature_matrix(ref_with_latent, mode="combined")
        y = ref_with_latent.obs["cell_type"].values
        clf, le = train_classifier(X, y, params={"n_estimators": 5, "max_depth": 2})

        bundle_dir = save_model_bundle(tmp_path / "bundle", clf, le)

        clf2, le2, scvi2 = load_model_bundle(bundle_dir, load_scvi=False)
        assert set(le2.classes_) == set(le.classes_)
        # Predictions should be identical
        import numpy as np
        from celllabeller.classifier import predict
        labels1, _ = predict(clf, X, label_encoder=le)
        labels2, _ = predict(clf2, X, label_encoder=le2)
        assert (labels1 == labels2).all()

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_bundle(tmp_path / "nonexistent", load_scvi=False)


class TestPipelinePersistence:
    def test_save_load_object(self, tmp_path):
        obj = _DummyPipeline()
        path = save_pipeline(obj, tmp_path / "pipe.joblib")
        loaded = load_pipeline(path)
        assert loaded.value == 42

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline(tmp_path / "missing.joblib")
