"""Tests for celllabeller.classifier."""
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from celllabeller.classifier import (
    hyperparameter_search,
    predict,
    train_classifier,
)
from celllabeller.features import build_feature_matrix


class TestTrainClassifier:
    def test_returns_clf_and_le(self, ref_with_latent):
        from xgboost import XGBClassifier

        X, _ = build_feature_matrix(ref_with_latent, mode="combined")
        y = ref_with_latent.obs["cell_type"].values
        clf, le = train_classifier(X, y, params={"n_estimators": 5, "max_depth": 2})
        assert isinstance(clf, XGBClassifier)
        assert isinstance(le, LabelEncoder)
        assert set(le.classes_) == set(np.unique(y))

    def test_predict_shape(self, ref_with_latent):
        X, _ = build_feature_matrix(ref_with_latent, mode="combined")
        y = ref_with_latent.obs["cell_type"].values
        clf, le = train_classifier(X, y, params={"n_estimators": 5, "max_depth": 2})
        labels, probs = predict(clf, X, label_encoder=le)
        assert labels.shape == (ref_with_latent.n_obs,)
        assert probs.shape == (ref_with_latent.n_obs, len(le.classes_))

    def test_labels_are_strings(self, ref_with_latent):
        X, _ = build_feature_matrix(ref_with_latent, mode="combined")
        y = ref_with_latent.obs["cell_type"].values
        clf, le = train_classifier(X, y, params={"n_estimators": 5, "max_depth": 2})
        labels, _ = predict(clf, X, label_encoder=le)
        assert all(isinstance(l, str) for l in labels)


class TestHyperparameterSearch:
    def test_returns_dict(self, ref_with_latent):
        X, _ = build_feature_matrix(ref_with_latent, mode="combined")
        y = ref_with_latent.obs["cell_type"].values
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        params = hyperparameter_search(
            X,
            y_enc,
            n_iter=3,
            cv=2,
            scoring="f1_weighted",
        )
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_best_params_used_in_training(self, ref_with_latent):
        X, _ = build_feature_matrix(ref_with_latent, mode="combined")
        y = ref_with_latent.obs["cell_type"].values
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        params = hyperparameter_search(X, y_enc, n_iter=2, cv=2)
        clf, _ = train_classifier(X, y, params=params)
        assert clf is not None
