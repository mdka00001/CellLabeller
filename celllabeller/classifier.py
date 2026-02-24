"""XGBoost cell-type classifier with hyperparameter search.

Workflow
--------
1. :func:`hyperparameter_search` – finds the best XGBoost hyperparameters
   via ``RandomizedSearchCV`` or an optional Optuna TPE study.
2. :func:`train_classifier`      – trains a final XGBoost model on the full
   reference feature matrix with the selected hyperparameters.
3. :func:`predict`               – generates class labels and probability
   scores for a query feature matrix.

Usage
-----
>>> from celllabeller.classifier import hyperparameter_search, train_classifier, predict
>>> best_params = hyperparameter_search(X_train, y_train)
>>> clf = train_classifier(X_train, y_train, params=best_params)
>>> labels, probs = predict(clf, X_query)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameter search space
# ---------------------------------------------------------------------------
DEFAULT_PARAM_DISTRIBUTIONS: Dict[str, Any] = {
    "n_estimators": [100, 200, 400, 600, 800],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.3, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0],
}


def _make_base_xgb(n_classes: int, seed: int = 42) -> XGBClassifier:
    """Return a base XGBClassifier with sensible defaults."""
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
    return XGBClassifier(
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        use_label_encoder=False,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
    )


def hyperparameter_search(
    X: np.ndarray,
    y: np.ndarray,
    *,
    param_distributions: Optional[Dict[str, Any]] = None,
    n_iter: int = 30,
    cv: int = 5,
    scoring: str = "f1_weighted",
    seed: int = 42,
    n_jobs: int = -1,
    use_optuna: bool = False,
    optuna_n_trials: int = 50,
) -> Dict[str, Any]:
    """Find the best XGBoost hyperparameters for the given training data.

    Parameters
    ----------
    X:
        Feature matrix, shape ``(n_cells, n_features)``.
    y:
        Integer-encoded cell-type labels, shape ``(n_cells,)``.
    param_distributions:
        Dictionary mapping parameter names to lists of candidate values.
        Defaults to :data:`DEFAULT_PARAM_DISTRIBUTIONS`.
    n_iter:
        Number of parameter settings sampled by ``RandomizedSearchCV``.
    cv:
        Number of stratified cross-validation folds.
    scoring:
        Sklearn scoring string (default ``"f1_weighted"``).
    seed:
        Random seed.
    n_jobs:
        Parallel jobs for CV (``-1`` = all CPUs).
    use_optuna:
        If ``True``, run an Optuna TPE study instead of random search.
    optuna_n_trials:
        Number of Optuna trials (only used when ``use_optuna=True``).

    Returns
    -------
    best_params : dict
        Best hyperparameter dictionary ready to pass to :func:`train_classifier`.
    """
    n_classes = len(np.unique(y))
    dist = param_distributions or DEFAULT_PARAM_DISTRIBUTIONS

    if use_optuna:
        return _optuna_search(
            X,
            y,
            n_classes=n_classes,
            n_trials=optuna_n_trials,
            cv=cv,
            scoring=scoring,
            seed=seed,
        )

    logger.info(
        "RandomizedSearchCV: n_iter=%d, cv=%d, scoring=%s", n_iter, cv, scoring
    )
    t0 = time.time()
    base = _make_base_xgb(n_classes, seed=seed)
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=dist,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=seed,
        refit=False,
        verbose=1,
    )
    search.fit(X, y)
    elapsed = time.time() - t0
    logger.info(
        "Best CV score (%.4f) found in %.1fs. Best params: %s",
        search.best_score_,
        elapsed,
        search.best_params_,
    )
    return search.best_params_


def _optuna_search(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    n_trials: int,
    cv: int,
    scoring: str,
    seed: int,
) -> Dict[str, Any]:
    """Internal Optuna-based hyperparameter search."""
    try:
        import optuna  # noqa: PLC0415
        from sklearn.model_selection import cross_val_score  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Optuna is not installed. Install it with: pip install optuna"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [100, 200, 400, 600, 800]
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
        }
        clf = _make_base_xgb(n_classes, seed=seed)
        clf.set_params(**params)
        cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, X, y, cv=cv_strat, scoring=scoring, n_jobs=-1)
        return float(scores.mean())

    logger.info("Optuna search: n_trials=%d, cv=%d, scoring=%s", n_trials, cv, scoring)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(
        "Best Optuna score (%.4f). Best params: %s",
        study.best_value,
        study.best_params,
    )
    return study.best_params


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: Optional[Dict[str, Any]] = None,
    label_encoder: Optional[LabelEncoder] = None,
    seed: int = 42,
) -> Tuple[XGBClassifier, LabelEncoder]:
    """Train an XGBoost classifier on *X* and *y*.

    Parameters
    ----------
    X:
        Feature matrix, shape ``(n_cells, n_features)``.
    y:
        Cell-type labels (strings or integers).
    params:
        Hyperparameter dictionary (e.g. from :func:`hyperparameter_search`).
        If ``None``, uses XGBoost defaults.
    label_encoder:
        Pre-fitted :class:`~sklearn.preprocessing.LabelEncoder`.  If
        ``None``, a new one is fitted on *y*.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    clf : XGBClassifier
        Trained classifier.
    le : LabelEncoder
        Fitted label encoder (maps string labels ↔ integers).
    """
    if label_encoder is None:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
    else:
        le = label_encoder
        y_enc = le.transform(y)

    n_classes = len(le.classes_)
    clf = _make_base_xgb(n_classes, seed=seed)

    if params:
        # Strip keys that are not XGBClassifier parameters
        valid_keys = set(clf.get_params().keys())
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        clf.set_params(**filtered)

    logger.info(
        "Training XGBClassifier on %d cells, %d features, %d classes.",
        X.shape[0],
        X.shape[1],
        n_classes,
    )
    t0 = time.time()
    clf.fit(X, y_enc)
    logger.info("Training completed in %.1fs.", time.time() - t0)
    return clf, le


def predict(
    clf: XGBClassifier,
    X: np.ndarray,
    label_encoder: Optional[LabelEncoder] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate cell-type predictions for *X*.

    Parameters
    ----------
    clf:
        Trained :class:`~xgboost.XGBClassifier`.
    X:
        Feature matrix, shape ``(n_cells, n_features)``.
    label_encoder:
        Fitted :class:`~sklearn.preprocessing.LabelEncoder` returned by
        :func:`train_classifier`.  If ``None``, returns integer class indices.

    Returns
    -------
    labels : np.ndarray, shape (n_cells,)
        Predicted cell-type labels (strings if *label_encoder* provided,
        otherwise integers).
    probs : np.ndarray, shape (n_cells, n_classes)
        Predicted probability for each class.
    """
    probs = clf.predict_proba(X)
    idx = np.argmax(probs, axis=1)

    if label_encoder is not None:
        labels = label_encoder.inverse_transform(idx)
    else:
        labels = idx

    return labels, probs
