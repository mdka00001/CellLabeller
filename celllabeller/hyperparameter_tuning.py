"""
XGBoost hyperparameter tuning module with GPU support
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Literal
from pathlib import Path
import logging
import warnings

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import optuna
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class XGBoostTuner:
    """
    Hyperparameter tuning for XGBoost classifier with GPU support.
    
    Parameters
    ----------
    X_features : np.ndarray
        Feature matrix (n_samples x n_features)
    y_labels : np.ndarray or list
        Cell type labels
    results_dir : Path or str
        Directory to save results and models
    test_size : float, default 0.2
        Proportion of data to use for testing
    random_state : int, default 42
        Random state for reproducibility
    """
    
    def __init__(
        self,
        X_features: np.ndarray,
        y_labels: np.ndarray,
        results_dir: Path,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.X_features = X_features
        self.y_labels = y_labels
        self.results_dir = Path(results_dir)
        self.test_size = test_size
        self.random_state = random_state
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(y_labels)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_features,
            self.y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_encoded,
        )
        
        self.best_models = {}  # Store models for different device configs
        self.tuning_results = {}
        self.trial_histories = {}
        
        logger.info(f"Training set: {self.X_train.shape[0]}, Test set: {self.X_test.shape[0]}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
    
    def tune_hyperparameters(
        self,
        use_gpu: bool = True,
        n_trials: int = 50,
        n_jobs: int = -1,
        verbose: int = 1,
    ) -> Dict:
        """
        Perform hyperparameter tuning using Optuna.
        
        Parameters
        ----------
        use_gpu : bool, default True
            Whether to use GPU for training
        n_trials : int, default 50
            Number of trials for hyperparameter optimization
        n_jobs : int, default -1
            Number of parallel jobs (-1 means use all processors)
        verbose : int, default 1
            Verbosity level for Optuna
        
        Returns
        -------
        Dict
            Best hyperparameters found
        """
        logger.info(f"Starting hyperparameter tuning with GPU={use_gpu}")
        device_key = f"gpu_{use_gpu}"
        
        # Define objective function
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
                "random_state": self.random_state,
                "tree_method": "gpu_hist" if use_gpu else "hist",
                "gpu_id": 0 if use_gpu else -1,
                "objective": "multi:softmax" if len(self.label_encoder.classes_) > 2 else "binary:logistic",
                "num_class": len(self.label_encoder.classes_) if len(self.label_encoder.classes_) > 2 else None,
                "eval_metric": "mlogloss" if len(self.label_encoder.classes_) > 2 else "logloss",
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            try:
                # Cross-validation on training set
                clf = xgb.XGBClassifier(**params)
                scores = cross_val_score(
                    clf, self.X_train, self.y_train, cv=5, scoring="balanced_accuracy"
                )
                
                return scores.mean()
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float("-inf")
        
        # Create study and optimize
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        
        logger.info(f"Running {n_trials} trials for GPU={use_gpu}...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=(verbose > 0))
        
        best_params = study.best_params
        logger.info(f"Best trial value: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Store results
        self.tuning_results[device_key] = {
            "best_params": best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
        }
        
        # Convert study to dataframe for analysis
        trials_df = study.trials_dataframe()
        self.trial_histories[device_key] = trials_df
        
        return best_params
    
    def train_best_model(
        self,
        use_gpu: bool = True,
        best_params: Optional[Dict] = None,
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model with best hyperparameters.
        
        Parameters
        ----------
        use_gpu : bool, default True
            Whether to use GPU for training
        best_params : Dict, optional
            Best parameters found from tuning. If None, runs tuning first
        
        Returns
        -------
        xgb.XGBClassifier
            Trained XGBoost classifier
        """
        device_key = f"gpu_{use_gpu}"
        
        if best_params is None:
            logger.info("No parameters provided, running tuning first...")
            best_params = self.tune_hyperparameters(use_gpu=use_gpu)
        
        # Prepare parameters for training
        training_params = best_params.copy()
        training_params.update({
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "gpu_id": 0 if use_gpu else -1,
            "objective": "multi:softmax" if len(self.label_encoder.classes_) > 2 else "binary:logistic",
            "eval_metric": "mlogloss" if len(self.label_encoder.classes_) > 2 else "logloss",
            "random_state": self.random_state,
        })
        
        if len(self.label_encoder.classes_) > 2:
            training_params["num_class"] = len(self.label_encoder.classes_)
        
        # Remove None values
        training_params = {k: v for k, v in training_params.items() if v is not None}
        
        logger.info(f"Training final model with GPU={use_gpu}...")
        model = xgb.XGBClassifier(**training_params)
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            early_stopping_rounds=10,
            verbose=False,
        )
        
        self.best_models[device_key] = model
        
        return model
    
    def evaluate_model(
        self,
        model: xgb.XGBClassifier,
        use_gpu: bool = True,
    ) -> Dict:
        """
        Evaluate trained model on both training and test sets.
        
        Parameters
        ----------
        model : xgb.XGBClassifier
            Trained XGBoost classifier
        use_gpu : bool, default True
            Label for results
        
        Returns
        -------
        Dict
            Performance metrics
        """
        device_key = f"gpu_{use_gpu}"
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        train_balanced_acc = balanced_accuracy_score(self.y_train, y_train_pred)
        test_balanced_acc = balanced_accuracy_score(self.y_test, y_test_pred)
        
        train_f1 = f1_score(self.y_train, y_train_pred, average="weighted", zero_division=0)
        test_f1 = f1_score(self.y_test, y_test_pred, average="weighted", zero_division=0)
        
        results = {
            "device": "GPU" if use_gpu else "CPU",
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_balanced_accuracy": train_balanced_acc,
            "test_balanced_accuracy": test_balanced_acc,
            "train_f1_weighted": train_f1,
            "test_f1_weighted": test_f1,
            "confusion_matrix": confusion_matrix(self.y_test, y_test_pred).tolist(),
            "classification_report": classification_report(
                self.y_test, y_test_pred, 
                target_names=self.label_encoder.classes_.tolist(),
                zero_division=0,
            ),
            "predictions_test": y_test_pred.tolist(),
            "predictions_train": y_train_pred.tolist(),
            "true_labels_test": self.y_test.tolist(),
            "true_labels_train": self.y_train.tolist(),
        }
        
        logger.info(f"\n{device_key.upper()} Performance:")
        logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Train Balanced Accuracy: {train_balanced_acc:.4f}")
        logger.info(f"  Test Balanced Accuracy: {test_balanced_acc:.4f}")
        
        return results
    
    def save_results(self, results_dict: Dict, use_gpu: bool = True):
        """
        Save tuning results and performance metrics to disk.
        
        Parameters
        ----------
        results_dict : Dict
            Results dictionary to save
        use_gpu : bool
            Whether GPU was used
        """
        device_key = f"gpu_{use_gpu}"
        device_str = "gpu" if use_gpu else "cpu"
        
        # Save evaluation results
        results_path = self.results_dir / f"evaluation_results_{device_str}.pkl"
        logger.info(f"Saving evaluation results to {results_path}")
        with open(results_path, "wb") as f:
            pickle.dump(results_dict, f)
        
        # Save tuning results
        if device_key in self.tuning_results:
            tuning_path = self.results_dir / f"tuning_results_{device_str}.pkl"
            logger.info(f"Saving tuning results to {tuning_path}")
            with open(tuning_path, "wb") as f:
                pickle.dump(self.tuning_results[device_key], f)
        
        # Save trial history as CSV
        if device_key in self.trial_histories:
            history_path = self.results_dir / f"trial_history_{device_str}.csv"
            logger.info(f"Saving trial history to {history_path}")
            self.trial_histories[device_key].to_csv(history_path, index=False)
    
    def save_model(self, model: xgb.XGBClassifier, use_gpu: bool = True):
        """
        Save trained model to disk.
        
        Parameters
        ----------
        model : xgb.XGBClassifier
            Model to save
        use_gpu : bool
            Whether GPU was used
        """
        device_str = "gpu" if use_gpu else "cpu"
        model_path = self.results_dir / f"xgboost_model_{device_str}.pkl"
        logger.info(f"Saving model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Also save label encoder
        encoder_path = self.results_dir / "label_encoder.pkl"
        logger.info(f"Saving label encoder to {encoder_path}")
        with open(encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)
    
    def compare_gpu_cpu(self) -> pd.DataFrame:
        """
        Run full pipeline for both GPU and CPU and compare results.
        
        Returns
        -------
        pd.DataFrame
            Comparison of GPU vs CPU results
        """
        logger.info("="*60)
        logger.info("STARTING GPU vs CPU COMPARISON")
        logger.info("="*60)
        
        all_results = {}
        
        for use_gpu in [True, False]:
            device_str = "GPU" if use_gpu else "CPU"
            logger.info(f"\n{device_str} CONFIGURATION")
            logger.info("-"*60)
            
            try:
                # Tune hyperparameters
                logger.info(f"Tuning hyperparameters on {device_str}...")
                best_params = self.tune_hyperparameters(use_gpu=use_gpu, n_trials=20)
                
                # Train model
                logger.info(f"Training model on {device_str}...")
                model = self.train_best_model(use_gpu=use_gpu, best_params=best_params)
                
                # Evaluate
                logger.info(f"Evaluating model on {device_str}...")
                results = self.evaluate_model(model, use_gpu=use_gpu)
                
                # Save
                self.save_model(model, use_gpu=use_gpu)
                self.save_results(results, use_gpu=use_gpu)
                
                all_results[device_str] = results
                
            except Exception as e:
                logger.error(f"Error during {device_str} processing: {e}")
                all_results[device_str] = {"error": str(e)}
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            device: [
                results.get("train_accuracy"),
                results.get("test_accuracy"),
                results.get("train_balanced_accuracy"),
                results.get("test_balanced_accuracy"),
            ]
            for device, results in all_results.items()
            if "error" not in results
        }, index=["Train Accuracy", "Test Accuracy", "Train Bal. Accuracy", "Test Bal. Accuracy"])
        
        # Save comparison
        comparison_path = self.results_dir / "gpu_cpu_comparison.csv"
        logger.info(f"Saving comparison to {comparison_path}")
        comparison_df.to_csv(comparison_path)
        
        logger.info("\nGPU vs CPU Comparison:")
        logger.info(comparison_df.to_string())
        
        return comparison_df
