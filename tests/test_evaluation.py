"""Tests for celllabeller.evaluation."""
import numpy as np
import pytest


class TestScorePredictions:
    def test_basic_metrics(self, trained_clf, ref_with_latent):
        from celllabeller.features import build_feature_matrix
        from celllabeller.classifier import predict
        from celllabeller.evaluation import score_predictions

        clf, le = trained_clf
        X, _ = build_feature_matrix(ref_with_latent, mode="combined")
        y_true = ref_with_latent.obs["cell_type"].values
        labels, probs = predict(clf, X, label_encoder=le)

        metrics = score_predictions(y_true, labels, probs, label_encoder=le)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_without_proba(self, trained_clf, ref_with_latent):
        from celllabeller.features import build_feature_matrix
        from celllabeller.classifier import predict
        from celllabeller.evaluation import score_predictions

        clf, le = trained_clf
        X, _ = build_feature_matrix(ref_with_latent, mode="combined")
        y_true = ref_with_latent.obs["cell_type"].values
        labels, _ = predict(clf, X, label_encoder=le)

        metrics = score_predictions(y_true, labels, label_encoder=le)
        assert "roc_auc" not in metrics


class TestPlotConfusionMatrix:
    def test_returns_figure(self, trained_clf, ref_with_latent):
        pytest.importorskip("matplotlib")
        from celllabeller.features import build_feature_matrix
        from celllabeller.classifier import predict
        from celllabeller.evaluation import plot_confusion_matrix
        import matplotlib.pyplot as plt

        clf, le = trained_clf
        X, _ = build_feature_matrix(ref_with_latent, mode="combined")
        y_true = ref_with_latent.obs["cell_type"].values
        labels, _ = predict(clf, X, label_encoder=le)

        fig = plot_confusion_matrix(y_true, labels, labels=le.classes_)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestPlotFeatureImportance:
    def test_returns_figure(self, trained_clf, ref_with_latent):
        pytest.importorskip("matplotlib")
        from celllabeller.features import build_feature_matrix
        from celllabeller.evaluation import plot_feature_importance
        import matplotlib.pyplot as plt

        clf, le = trained_clf
        _, feature_names = build_feature_matrix(ref_with_latent, mode="combined")

        fig = plot_feature_importance(clf, feature_names, top_n=10)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
