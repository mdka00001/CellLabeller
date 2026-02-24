"""High-level ``CellLabeller`` pipeline.

Orchestrates the full workflow:

1. Integrate reference + query AnnData objects via scVI.
2. Build feature matrices (gene expression, scVI latent, or combined).
3. Perform hyperparameter search on XGBoost.
4. Train the final XGBoost cell-type classifier on the reference.
5. Transfer predictions to the query dataset.
6. Score and plot results.
7. Save / load the full pipeline.

Minimal usage
-------------
>>> from celllabeller import CellLabeller
>>> cl = CellLabeller(celltype_key="cell_type")
>>> cl.fit(ref_adata, query_adata)
>>> cl.label_query()
>>> cl.score(query_adata)          # if ground-truth labels are available
>>> cl.save("my_model_bundle")
>>> # Later …
>>> cl2 = CellLabeller.load("my_model_bundle")
>>> cl2.label_query(new_query_adata)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import anndata as ad
import numpy as np

from .classifier import hyperparameter_search, predict, train_classifier
from .evaluation import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_prediction_confidence,
    plot_umap,
    score_predictions,
)
from .features import build_feature_matrix
from .integration import integrate_datasets
from .models import load_model_bundle, load_pipeline, save_model_bundle, save_pipeline

logger = logging.getLogger(__name__)


class CellLabeller:
    """End-to-end single-cell annotation pipeline.

    Parameters
    ----------
    celltype_key:
        Key in ``ref_adata.obs`` that holds ground-truth cell-type labels.
    feature_mode:
        Feature engineering strategy: ``"gene"``, ``"latent"``, or
        ``"combined"`` (default).
    n_latent:
        Dimensionality of the scVI latent space.
    n_layers:
        Number of hidden layers in the scVI encoder/decoder.
    n_hidden:
        Number of units per hidden layer in scVI.
    max_epochs_scvi:
        Maximum scVI training epochs.
    hyperparameter_search:
        If ``True``, run XGBoost hyperparameter search before final training.
    hp_n_iter:
        Number of iterations for ``RandomizedSearchCV``.
    hp_cv:
        Cross-validation folds for hyperparameter search.
    hp_scoring:
        Scoring metric for hyperparameter selection.
    use_optuna:
        Use Optuna TPE instead of ``RandomizedSearchCV``.
    optuna_n_trials:
        Number of Optuna trials.
    seed:
        Global random seed.
    accelerator:
        PyTorch-Lightning accelerator (``"cpu"``, ``"gpu"``, …).
    """

    def __init__(
        self,
        *,
        celltype_key: str = "cell_type",
        feature_mode: str = "combined",
        n_latent: int = 30,
        n_layers: int = 2,
        n_hidden: int = 128,
        max_epochs_scvi: int = 400,
        hyperparameter_search: bool = True,
        hp_n_iter: int = 30,
        hp_cv: int = 5,
        hp_scoring: str = "f1_weighted",
        use_optuna: bool = False,
        optuna_n_trials: int = 50,
        seed: int = 42,
        accelerator: str = "cpu",
    ):
        self.celltype_key = celltype_key
        self.feature_mode = feature_mode
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.max_epochs_scvi = max_epochs_scvi
        self.run_hyperparameter_search = hyperparameter_search
        self.hp_n_iter = hp_n_iter
        self.hp_cv = hp_cv
        self.hp_scoring = hp_scoring
        self.use_optuna = use_optuna
        self.optuna_n_trials = optuna_n_trials
        self.seed = seed
        self.accelerator = accelerator

        # Set during fit
        self.ref_adata_: Optional[ad.AnnData] = None
        self.query_adata_: Optional[ad.AnnData] = None
        self.scvi_model_ = None
        self.clf_ = None
        self.label_encoder_ = None
        self.feature_names_: Optional[list[str]] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.common_genes_: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        ref_adata: ad.AnnData,
        query_adata: ad.AnnData,
        *,
        batch_key: Optional[str] = None,
    ) -> "CellLabeller":
        """Integrate datasets and train the cell-type classifier.

        Parameters
        ----------
        ref_adata:
            Reference AnnData (raw counts in ``X``, cell-type labels in
            ``obs[celltype_key]``).
        query_adata:
            Query AnnData (raw counts in ``X``).
        batch_key:
            obs column to use as a scVI batch covariate (optional).

        Returns
        -------
        self
        """
        if self.celltype_key not in ref_adata.obs.columns:
            raise ValueError(
                f"celltype_key '{self.celltype_key}' not found in ref_adata.obs. "
                f"Available columns: {list(ref_adata.obs.columns)}"
            )

        # 1. Integrate
        logger.info("Step 1/4 – scVI integration …")
        ref_sub, query_sub, scvi_model = integrate_datasets(
            ref_adata,
            query_adata,
            n_latent=self.n_latent,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            max_epochs=self.max_epochs_scvi,
            batch_key=batch_key,
            seed=self.seed,
            accelerator=self.accelerator,
        )
        self.ref_adata_ = ref_sub
        self.query_adata_ = query_sub
        self.scvi_model_ = scvi_model
        self.common_genes_ = ref_sub.var_names.tolist()

        # 2. Feature engineering
        logger.info("Step 2/4 – Feature engineering (mode=%s) …", self.feature_mode)
        X_ref, feature_names = build_feature_matrix(ref_sub, mode=self.feature_mode)
        self.feature_names_ = feature_names
        y_ref = ref_sub.obs[self.celltype_key].values

        # 3. Hyperparameter search
        if self.run_hyperparameter_search:
            logger.info("Step 3/4 – Hyperparameter search …")
            # Local import: LabelEncoder is only needed here for a temporary
            # integer encoding required by hyperparameter_search(); the final
            # encoder used throughout the pipeline is created inside
            # train_classifier() below.
            from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415

            le_tmp = LabelEncoder()
            y_enc = le_tmp.fit_transform(y_ref)
            self.best_params_ = hyperparameter_search(
                X_ref,
                y_enc,
                n_iter=self.hp_n_iter,
                cv=self.hp_cv,
                scoring=self.hp_scoring,
                seed=self.seed,
                use_optuna=self.use_optuna,
                optuna_n_trials=self.optuna_n_trials,
            )
            logger.info("Best params: %s", self.best_params_)
        else:
            logger.info("Step 3/4 – Skipping hyperparameter search.")

        # 4. Train final classifier
        logger.info("Step 4/4 – Training final XGBoost classifier …")
        self.clf_, self.label_encoder_ = train_classifier(
            X_ref,
            y_ref,
            params=self.best_params_,
            seed=self.seed,
        )
        logger.info("CellLabeller.fit() complete.")
        return self

    def label_query(
        self,
        query_adata: Optional[ad.AnnData] = None,
    ) -> ad.AnnData:
        """Transfer cell-type labels to a query AnnData.

        If *query_adata* is ``None``, the query dataset stored during
        :meth:`fit` is used.

        Predictions are stored in:
        * ``obs["predicted_celltype"]`` – string label.
        * ``obs["prediction_confidence"]`` – max class probability.
        * ``obsm["prediction_proba"]`` – full probability matrix.

        Parameters
        ----------
        query_adata:
            New query AnnData to annotate (raw counts, same genes as
            training set).  If ``None``, uses ``self.query_adata_``.

        Returns
        -------
        query_adata : ad.AnnData
            The annotated query AnnData (in-place modification).
        """
        self._check_fitted()

        if query_adata is None:
            query_adata = self.query_adata_
        else:
            # Subset to common genes and build latent if needed
            query_adata = self._prepare_new_query(query_adata)

        X_query, _ = build_feature_matrix(query_adata, mode=self.feature_mode)
        labels, probs = predict(self.clf_, X_query, label_encoder=self.label_encoder_)

        query_adata.obs["predicted_celltype"] = labels
        query_adata.obs["prediction_confidence"] = np.max(probs, axis=1)
        query_adata.obsm["prediction_proba"] = probs

        logger.info(
            "Labelled %d query cells (%d unique predicted types).",
            query_adata.n_obs,
            len(np.unique(labels)),
        )
        return query_adata

    def score(
        self,
        query_adata: ad.AnnData,
        *,
        true_label_key: Optional[str] = None,
        average: str = "weighted",
    ) -> dict:
        """Score predictions on *query_adata*.

        Parameters
        ----------
        query_adata:
            AnnData with ``obs["predicted_celltype"]`` already populated
            (run :meth:`label_query` first) and optionally ground-truth
            labels in ``obs[true_label_key]``.
        true_label_key:
            obs column with ground-truth cell types.  Defaults to
            ``self.celltype_key``.
        average:
            Averaging strategy for precision/recall/F1.

        Returns
        -------
        metrics : dict
        """
        self._check_fitted()
        key = true_label_key or self.celltype_key
        if key not in query_adata.obs.columns:
            raise KeyError(
                f"Ground-truth key '{key}' not found in query_adata.obs."
            )
        if "predicted_celltype" not in query_adata.obs.columns:
            raise KeyError(
                "Run label_query() before score()."
            )

        y_true = query_adata.obs[key].values
        y_pred = query_adata.obs["predicted_celltype"].values
        y_proba = query_adata.obsm.get("prediction_proba")

        return score_predictions(
            y_true,
            y_pred,
            y_proba,
            label_encoder=self.label_encoder_,
            average=average,
        )

    # ------------------------------------------------------------------
    # Plotting convenience methods
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self, query_adata: ad.AnnData, **kwargs):
        """Plot confusion matrix comparing predictions to ground truth."""
        self._check_fitted()
        key = self.celltype_key
        y_true = query_adata.obs[key].values
        y_pred = query_adata.obs["predicted_celltype"].values
        labels = self.label_encoder_.classes_
        return plot_confusion_matrix(y_true, y_pred, labels=labels, **kwargs)

    def plot_umap(self, adata: Optional[ad.AnnData] = None, **kwargs):
        """Plot UMAP of *adata* coloured by predicted cell types."""
        self._check_fitted()
        adata = adata if adata is not None else self.query_adata_
        return plot_umap(adata, **kwargs)

    def plot_prediction_confidence(
        self, adata: Optional[ad.AnnData] = None, **kwargs
    ):
        """Violin plot of prediction confidence per cell type."""
        self._check_fitted()
        adata = adata if adata is not None else self.query_adata_
        return plot_prediction_confidence(adata, **kwargs)

    def plot_feature_importance(self, **kwargs):
        """Bar chart of top XGBoost feature importances."""
        self._check_fitted()
        return plot_feature_importance(self.clf_, self.feature_names_, **kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path, *, save_full_pipeline: bool = True) -> Path:
        """Save the trained model bundle.

        Parameters
        ----------
        output_dir:
            Target directory for the bundle.
        save_full_pipeline:
            Also serialise this ``CellLabeller`` instance as a ``joblib``
            file inside *output_dir* for easy reuse with :meth:`load`.

        Returns
        -------
        Path
            Absolute path to the bundle directory.
        """
        self._check_fitted()
        bundle_dir = save_model_bundle(
            output_dir,
            self.clf_,
            self.label_encoder_,
            self.scvi_model_,
            extra_meta={
                "celltype_key": self.celltype_key,
                "feature_mode": self.feature_mode,
                "n_latent": self.n_latent,
                "common_genes": self.common_genes_,
                "best_params": self.best_params_,
                "feature_names": self.feature_names_,
            },
        )
        if save_full_pipeline:
            save_pipeline(self, Path(bundle_dir) / "pipeline.joblib")
        logger.info("CellLabeller saved to %s", bundle_dir)
        return bundle_dir

    @classmethod
    def load(
        cls,
        bundle_dir: str | Path,
        *,
        ref_adata: Optional[ad.AnnData] = None,
    ) -> "CellLabeller":
        """Load a previously saved ``CellLabeller`` pipeline.

        Prefers the full ``pipeline.joblib`` if it exists; otherwise
        reconstructs from the model bundle artefacts.

        Parameters
        ----------
        bundle_dir:
            Path to the bundle directory (same as passed to :meth:`save`).
        ref_adata:
            Reference AnnData needed to reload the scVI model.

        Returns
        -------
        CellLabeller
        """
        bundle_dir = Path(bundle_dir)
        pipeline_file = bundle_dir / "pipeline.joblib"
        if pipeline_file.exists():
            obj = load_pipeline(pipeline_file)
            if isinstance(obj, cls):
                return obj
            logger.warning(
                "Loaded pipeline.joblib is not a CellLabeller; "
                "falling back to model bundle."
            )

        import joblib  # noqa: PLC0415

        clf, le, scvi_model = load_model_bundle(
            bundle_dir, load_scvi=True, ref_adata=ref_adata
        )
        meta_path = bundle_dir / "bundle_meta.joblib"
        meta = joblib.load(meta_path) if meta_path.exists() else {}

        instance = cls(
            celltype_key=meta.get("celltype_key", "cell_type"),
            feature_mode=meta.get("feature_mode", "combined"),
            n_latent=meta.get("n_latent", 30),
        )
        instance.clf_ = clf
        instance.label_encoder_ = le
        instance.scvi_model_ = scvi_model
        instance.common_genes_ = meta.get("common_genes")
        instance.best_params_ = meta.get("best_params")
        instance.feature_names_ = meta.get("feature_names")
        return instance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        """Raise if the model has not been trained yet."""
        if self.clf_ is None or self.label_encoder_ is None:
            raise RuntimeError(
                "CellLabeller is not fitted yet. Call fit() first."
            )

    def _prepare_new_query(self, query_adata: ad.AnnData) -> ad.AnnData:
        """Subset *query_adata* to common genes and project into latent space."""
        import scvi  # noqa: PLC0415

        common = self.common_genes_
        missing = set(common) - set(query_adata.var_names)
        if missing:
            raise ValueError(
                f"{len(missing)} genes required by the model are missing from "
                f"the new query dataset.  First missing: {sorted(missing)[:5]}"
            )
        query_sub = query_adata[:, common].copy()

        scvi.model.SCVI.prepare_query_anndata(query_sub, self.scvi_model_)
        query_model = scvi.model.SCVI.load_query_data(query_sub, self.scvi_model_)
        query_model.train(
            max_epochs=200,
            plan_kwargs={"weight_decay": 0.0},
            early_stopping=True,
            accelerator=self.accelerator,
        )
        query_sub.obsm["X_scVI"] = query_model.get_latent_representation()
        return query_sub

    def __repr__(self) -> str:
        fitted = self.clf_ is not None
        return (
            f"CellLabeller("
            f"feature_mode={self.feature_mode!r}, "
            f"n_latent={self.n_latent}, "
            f"fitted={fitted})"
        )

    # ------------------------------------------------------------------
    # Pickle support  (scvi_model_ is saved separately via scvi.save)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # scvi_model_ is persisted via scvi.model.SCVI.save() inside
        # save_model_bundle(); drop it here to keep the joblib artefact
        # portable and free of unpicklable scVI internals.
        state.pop("scvi_model_", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Re-initialise scvi_model_ as None; callers should use load()
        # which reloads it from the scVI sub-directory.
        if "scvi_model_" not in self.__dict__:
            self.scvi_model_ = None
