"""Model persistence utilities for CellLabeller.

Supports saving and loading:
* The trained :class:`~xgboost.XGBClassifier`
* The fitted :class:`~sklearn.preprocessing.LabelEncoder`
* The trained :class:`~scvi.model.SCVI` reference model
* The full ``CellLabeller`` pipeline state (via :mod:`joblib`)

Usage
-----
>>> from celllabeller.models import save_model_bundle, load_model_bundle
>>> save_model_bundle(output_dir, clf, label_encoder, scvi_model)
>>> clf, le, scvi_model = load_model_bundle(output_dir)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# File-name constants inside a bundle directory
_XGB_FNAME = "xgb_classifier.ubj"
_LE_FNAME = "label_encoder.joblib"
_SCVI_SUBDIR = "scvi_model"
_META_FNAME = "bundle_meta.joblib"


def save_model_bundle(
    output_dir: str | Path,
    clf: XGBClassifier,
    label_encoder: LabelEncoder,
    scvi_model=None,
    *,
    extra_meta: Optional[dict] = None,
) -> Path:
    """Save a complete model bundle to *output_dir*.

    Parameters
    ----------
    output_dir:
        Directory to write all artefacts into (created if absent).
    clf:
        Trained :class:`~xgboost.XGBClassifier`.
    label_encoder:
        Fitted :class:`~sklearn.preprocessing.LabelEncoder`.
    scvi_model:
        Trained :class:`scvi.model.SCVI` instance, or ``None``.
    extra_meta:
        Optional dictionary of arbitrary metadata to persist alongside
        the model (e.g. ``{"feature_mode": "combined", "n_latent": 30}``).

    Returns
    -------
    bundle_dir : Path
        Absolute path to the bundle directory.
    """
    bundle_dir = Path(output_dir).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # XGBoost — native binary format
    xgb_path = bundle_dir / _XGB_FNAME
    clf.save_model(str(xgb_path))
    logger.info("XGBClassifier saved to %s", xgb_path)

    # LabelEncoder — joblib
    le_path = bundle_dir / _LE_FNAME
    joblib.dump(label_encoder, le_path)
    logger.info("LabelEncoder saved to %s", le_path)

    # scVI model
    if scvi_model is not None:
        scvi_dir = bundle_dir / _SCVI_SUBDIR
        scvi_model.save(str(scvi_dir), overwrite=True)
        logger.info("scVI model saved to %s", scvi_dir)

    # Metadata
    meta = extra_meta or {}
    meta["classes_"] = label_encoder.classes_.tolist()
    joblib.dump(meta, bundle_dir / _META_FNAME)

    logger.info("Model bundle saved to %s", bundle_dir)
    return bundle_dir


def load_model_bundle(
    bundle_dir: str | Path,
    *,
    load_scvi: bool = True,
    ref_adata=None,
) -> Tuple[XGBClassifier, LabelEncoder, Optional[object]]:
    """Load a model bundle previously saved with :func:`save_model_bundle`.

    Parameters
    ----------
    bundle_dir:
        Path to the bundle directory.
    load_scvi:
        Whether to attempt loading the scVI model sub-directory.
    ref_adata:
        Reference AnnData required by ``scvi.model.SCVI.load()`` (pass the
        same object used during training or any subset with the same genes).

    Returns
    -------
    clf : XGBClassifier
    label_encoder : LabelEncoder
    scvi_model : scvi.model.SCVI or None
    """
    bundle_dir = Path(bundle_dir).resolve()

    # XGBoost
    xgb_path = bundle_dir / _XGB_FNAME
    if not xgb_path.exists():
        raise FileNotFoundError(f"XGBClassifier file not found: {xgb_path}")
    clf = XGBClassifier()
    clf.load_model(str(xgb_path))
    logger.info("XGBClassifier loaded from %s", xgb_path)

    # LabelEncoder
    le_path = bundle_dir / _LE_FNAME
    if not le_path.exists():
        raise FileNotFoundError(f"LabelEncoder file not found: {le_path}")
    label_encoder: LabelEncoder = joblib.load(le_path)
    logger.info("LabelEncoder loaded from %s", le_path)

    # scVI
    scvi_model = None
    scvi_dir = bundle_dir / _SCVI_SUBDIR
    if load_scvi and scvi_dir.exists():
        try:
            import scvi  # noqa: PLC0415

            scvi_model = scvi.model.SCVI.load(str(scvi_dir), adata=ref_adata)
            logger.info("scVI model loaded from %s", scvi_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load scVI model: %s", exc)

    return clf, label_encoder, scvi_model


def save_pipeline(pipeline_obj, output_path: str | Path) -> Path:
    """Persist an arbitrary Python object (e.g. :class:`~celllabeller.pipeline.CellLabeller`)
    using :mod:`joblib`.

    Parameters
    ----------
    pipeline_obj:
        Object to serialise.
    output_path:
        Destination ``.joblib`` file path.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_obj, output_path)
    logger.info("Pipeline object saved to %s", output_path)
    return output_path


def load_pipeline(input_path: str | Path):
    """Load a previously serialised pipeline object.

    Parameters
    ----------
    input_path:
        Path to the ``.joblib`` file.

    Returns
    -------
    object
        The deserialised pipeline object.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {input_path}")
    obj = joblib.load(input_path)
    logger.info("Pipeline object loaded from %s", input_path)
    return obj
