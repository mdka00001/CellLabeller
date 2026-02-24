"""
Utility functions for CellLabeller
"""

import pickle
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def load_model(model_path: Path) -> object:
    """
    Load a saved XGBoost model.
    
    Parameters
    ----------
    model_path : Path
        Path to the saved model
    
    Returns
    -------
    object
        Loaded XGBoost classifier
    """
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_label_encoder(encoder_path: Path) -> object:
    """
    Load a saved label encoder.
    
    Parameters
    ----------
    encoder_path : Path
        Path to the saved encoder
    
    Returns
    -------
    object
        Loaded LabelEncoder
    """
    logger.info(f"Loading label encoder from {encoder_path}")
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    return encoder


def load_results(results_path: Path) -> Dict:
    """
    Load saved results pickle file.
    
    Parameters
    ----------
    results_path : Path
        Path to the results pickle file
    
    Returns
    -------
    Dict
        Loaded results
    """
    logger.info(f"Loading results from {results_path}")
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    return results


def predict_cell_types(
    query_features: np.ndarray,
    model_path: Path,
    encoder_path: Path,
) -> np.ndarray:
    """
    Predict cell types for query features.
    
    Parameters
    ----------
    query_features : np.ndarray
        Feature matrix for query cells
    model_path : Path
        Path to trained model
    encoder_path : Path
        Path to label encoder
    
    Returns
    -------
    np.ndarray
        Predicted cell type labels
    """
    model = load_model(model_path)
    encoder = load_label_encoder(encoder_path)
    
    # Predict
    y_pred_encoded = model.predict(query_features)
    y_pred_labels = encoder.inverse_transform(y_pred_encoded)
    
    return y_pred_labels


def get_prediction_probabilities(
    query_features: np.ndarray,
    model_path: Path,
) -> np.ndarray:
    """
    Get prediction probabilities for query features.
    
    Parameters
    ----------
    query_features : np.ndarray
        Feature matrix for query cells
    model_path : Path
        Path to trained model
    
    Returns
    -------
    np.ndarray
        Prediction probabilities (n_samples x n_classes)
    """
    model = load_model(model_path)
    proba = model.predict_proba(query_features)
    return proba


def summarize_results(results_dir: Path) -> pd.DataFrame:
    """
    Summarize all results from a results directory.
    
    Parameters
    ----------
    results_dir : Path
        Path to results directory
    
    Returns
    -------
    pd.DataFrame
        Summary of all results
    """
    summary_data = {}
    
    # Load evaluation results
    for results_file in results_dir.glob("evaluation_results_*.pkl"):
        device = results_file.stem.split("_")[-1]
        results = load_results(results_file)
        summary_data[device] = {
            "train_accuracy": results.get("train_accuracy"),
            "test_accuracy": results.get("test_accuracy"),
            "train_balanced_accuracy": results.get("train_balanced_accuracy"),
            "test_balanced_accuracy": results.get("test_balanced_accuracy"),
            "train_f1": results.get("train_f1_weighted"),
            "test_f1": results.get("test_f1_weighted"),
        }
    
    summary_df = pd.DataFrame(summary_data).T
    return summary_df
