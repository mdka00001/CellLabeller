# CellLabeller API Documentation

## Module Overview

CellLabeller consists of three main modules for end-to-end cell type label transfer:

1. **label_transfer.py** - Core integration and preprocessing
2. **feature_engineering.py** - Feature selection and preparation
3. **hyperparameter_tuning.py** - Model optimization and evaluation
4. **utils.py** - Utility functions for model loading and prediction

---

## Module: label_transfer

### Class: `CellTypeLabelTransfer`

Main orchestrator for preprocessing and scVI integration.

#### Initialization

```python
CellTypeLabelTransfer(
    reference_adata: ad.AnnData,
    query_adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    batch_key: Optional[str] = None,
    results_dir: str = "./celllabeller_results",
    n_epochs: int = 200,
    device: str = "gpu",
)
```

**Parameters:**
- `reference_adata`: Reference AnnData object with cell type annotations in `.obs`
- `query_adata`: Query AnnData object to be labeled
- `cell_type_key`: Column name in `reference_adata.obs` containing cell type labels
- `batch_key`: Optional column name for batch correction in scVI (e.g., "batch", "donor")
- `results_dir`: Directory where all results and models will be saved
- `n_epochs`: Number of scVI training epochs (enforced minimum: 200)
- `device`: Computing device ("gpu" or "cpu")

**Attributes:**
- `reference_adata`: Processed reference dataset
- `query_adata`: Processed query dataset
- `integrated_adata`: Integrated reference and query (after integration)

#### Methods

**`subset_common_genes() -> Tuple[ad.AnnData, ad.AnnData]`**

Identifies and subsets both datasets to genes present in both.

```python
adata_ref, adata_query = label_transfer.subset_common_genes()
```

Returns:
- Reference AnnData with common genes only
- Query AnnData with common genes only

**`integrate_with_scvi() -> ad.AnnData`**

Integrates reference and query datasets using scVI.

```python
adata_integrated = label_transfer.integrate_with_scvi()
```

Returns:
- Integrated AnnData object with:
  - `.obsm["X_scvi"]`: scVI latent representation
  - `.obs["dataset"]`: "reference" or "query" label
  - Normalized and log-transformed expression

**`get_results_dir() -> Path`**

Get the results directory path.

```python
results_path = label_transfer.get_results_dir()
```

Returns: `pathlib.Path` object

**`get_integrated_adata() -> ad.AnnData`**

Retrieve the integrated AnnData object.

```python
integrated = label_transfer.get_integrated_adata()
```

Returns: Integrated AnnData object (or raises ValueError if not yet integrated)

**`save_integrated_data(filename: str = "integrated_data.h5ad")`**

Save the integrated data to disk.

```python
label_transfer.save_integrated_data("my_integrated_data.h5ad")
```

---

## Module: feature_engineering

### Class: `FeatureEngineer`

Flexible feature selection from genes and/or scVI latent space.

#### Initialization

```python
FeatureEngineer(
    integrated_adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    n_features_genes: int = 500,
)
```

**Parameters:**
- `integrated_adata`: Integrated AnnData (output from `label_transfer.integrate_with_scvi()`)
- `cell_type_key`: Column name in `.obs` with cell type labels
- `n_features_genes`: Number of top genes to select using f-score ranking

#### Methods

**`select_features(feature_type: str = "combined", n_scvi_components: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, list]`**

Select features based on user preference.

```python
X_features, indices, feature_names = feature_engineer.select_features(
    feature_type="combined",
    n_scvi_components=30
)
```

**Parameters:**
- `feature_type`: One of:
  - `"genes"`: Top N genes selected by f-score (for interpretability)
  - `"scvi_latent"`: scVI latent space only (for integration-driven classification)
  - `"combined"`: Both genes and latent space concatenated (best overall)
- `n_scvi_components`: Optional limit on number of scVI latent dimensions (default: all)

Returns:
- `X_features`: Feature matrix (n_cells × n_features), standardized
- `indices`: Cell indices
- `feature_names`: List of feature identifiers

**`get_feature_importance_genes() -> pd.DataFrame`**

Calculate feature importance scores for selected genes.

```python
importance_df = feature_engineer.get_feature_importance_genes()
# Returns: DataFrame with columns ["feature", "importance"]
```

---

## Module: hyperparameter_tuning

### Class: `XGBoostTuner`

Hyperparameter optimization for XGBoost with GPU/CPU support.

#### Initialization

```python
XGBoostTuner(
    X_features: np.ndarray,
    y_labels: np.ndarray,
    results_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
)
```

**Parameters:**
- `X_features`: Feature matrix (n_cells × n_features)
- `y_labels`: Cell type labels (n_cells,)
- `results_dir`: Directory for saving results
- `test_size`: Fraction of data for testing (default: 0.2)
- `random_state`: Random seed for reproducibility

**Attributes:**
- `X_train`, `X_test`: Training and test feature matrices
- `y_train`, `y_test`: Training and test labels (encoded)
- `label_encoder`: LabelEncoder for mapping predictions to original labels

#### Methods

**`tune_hyperparameters(use_gpu: bool = True, n_trials: int = 50, n_jobs: int = -1, verbose: int = 1) -> Dict`**

Perform Bayesian hyperparameter optimization using Optuna.

```python
best_params = tuner.tune_hyperparameters(
    use_gpu=True,
    n_trials=50,
    n_jobs=-1
)
```

**Parameters:**
- `use_gpu`: Use GPU acceleration for XGBoost training
- `n_trials`: Number of optimization trials (more = better but slower)
- `n_jobs`: Number of parallel jobs (-1 = use all CPU cores)
- `verbose`: Verbosity level for progress output

Returns: Dictionary with best hyperparameters found

**Hyperparameters Optimized:**
- `n_estimators`: [100, 1000]
- `max_depth`: [3, 12]
- `learning_rate`: [1e-4, 1e-1] (log-scale)
- `subsample`: [0.6, 1.0]
- `colsample_bytree`: [0.6, 1.0]
- `gamma`: [0, 10]
- `min_child_weight`: [1, 10]
- `reg_lambda`: [1e-5, 10] (log-scale)
- `reg_alpha`: [1e-5, 10] (log-scale)

**`train_best_model(use_gpu: bool = True, best_params: Optional[Dict] = None) -> xgb.XGBClassifier`**

Train XGBoost model with best hyperparameters.

```python
model = tuner.train_best_model(use_gpu=True, best_params=best_params)
```

Returns: Trained XGBClassifier object

**`evaluate_model(model: xgb.XGBClassifier, use_gpu: bool = True) -> Dict`**

Evaluate model on training and test sets.

```python
results = tuner.evaluate_model(model, use_gpu=True)
```

Returns: Dictionary with metrics:
- `train_accuracy`: Accuracy on training set
- `test_accuracy`: Accuracy on test set
- `train_balanced_accuracy`: Balanced accuracy on training set
- `test_balanced_accuracy`: Balanced accuracy on test set
- `train_f1_weighted`: F1-score (weighted) on training set
- `test_f1_weighted`: F1-score (weighted) on test set
- `confusion_matrix`: Confusion matrix
- `classification_report`: Detailed classification metrics
- `predictions_train`, `predictions_test`: Predicted labels
- `true_labels_train`, `true_labels_test`: True labels

**`save_model(model: xgb.XGBClassifier, use_gpu: bool = True)`**

Save trained model and label encoder.

```python
tuner.save_model(model, use_gpu=True)
# Saves: xgboost_model_gpu.pkl, label_encoder.pkl
```

**`save_results(results_dict: Dict, use_gpu: bool = True)`**

Save evaluation results and trial history.

```python
tuner.save_results(results, use_gpu=True)
# Saves: evaluation_results_gpu.pkl, tuning_results_gpu.pkl, trial_history_gpu.csv
```

**`compare_gpu_cpu() -> pd.DataFrame`**

Run full pipeline for both GPU and CPU and compare results.

```python
comparison_df = tuner.compare_gpu_cpu()
print(comparison_df)
```

Returns: DataFrame comparing GPU vs CPU performance across metrics

---

## Module: utils

Utility functions for loading and using trained models.

**`load_model(model_path: Path) -> object`**

Load a saved XGBoost model.

```python
from celllabeller.utils import load_model
model = load_model("results/xgboost_model_cpu.pkl")
```

**`load_label_encoder(encoder_path: Path) -> object`**

Load a saved label encoder.

```python
from celllabeller.utils import load_label_encoder
encoder = load_label_encoder("results/label_encoder.pkl")
```

**`predict_cell_types(query_features: np.ndarray, model_path: Path, encoder_path: Path) -> np.ndarray`**

Convenience function for prediction on new data.

```python
predictions = predict_cell_types(new_features, model_path, encoder_path)
```

Returns: Array of predicted cell type labels

**`get_prediction_probabilities(query_features: np.ndarray, model_path: Path) -> np.ndarray`**

Get prediction probabilities for each class.

```python
probabilities = get_prediction_probabilities(new_features, model_path)
# Returns: (n_samples, n_classes) array
```

**`summarize_results(results_dir: Path) -> pd.DataFrame`**

Summarize all results in a directory.

```python
summary = summarize_results("results/")
print(summary)
```

Returns: DataFrame with performance metrics for all models

---

## Typical Workflow Example

```python
import anndata as ad
from celllabeller import CellTypeLabelTransfer, FeatureEngineer, XGBoostTuner

# 1. Load and initialize
adata_ref = ad.read_h5ad("reference.h5ad")
adata_query = ad.read_h5ad("query.h5ad")

label_transfer = CellTypeLabelTransfer(
    reference_adata=adata_ref,
    query_adata=adata_query,
    cell_type_key="cell_type",
    n_epochs=250,
)

# 2. Preprocessing
label_transfer.subset_common_genes()
adata_integrated = label_transfer.integrate_with_scvi()

# 3. Feature engineering
fe = FeatureEngineer(adata_integrated)
X_features, _, feature_names = fe.select_features(feature_type="combined")

# 4. Model optimization
tuner = XGBoostTuner(
    X_features=X_features,
    y_labels=adata_integrated.obs["cell_type"].values,
    results_dir=label_transfer.get_results_dir(),
)

best_params = tuner.tune_hyperparameters(use_gpu=True, n_trials=50)
model = tuner.train_best_model(use_gpu=True, best_params=best_params)
results = tuner.evaluate_model(model, use_gpu=True)

# 5. Save results
tuner.save_model(model, use_gpu=True)
tuner.save_results(results, use_gpu=True)

# 6. Make predictions
ref_mask = adata_integrated.obs["dataset"] == "query"
X_query = X_features[ref_mask]
y_pred = model.predict(X_query)
y_proba = model.predict_proba(X_query)
```

---

## Error Handling

### Common Issues and Solutions

**GPU Not Available:**
```python
# Use CPU instead
label_transfer = CellTypeLabelTransfer(..., device="cpu")
```

**Memory Error During Integration:**
```python
# Reduce feature set or use smaller batch size
fe = FeatureEngineer(adata_integrated, n_features_genes=300)
```

**Hyperparameter Tuning Too Slow:**
```python
# Reduce number of trials
best_params = tuner.tune_hyperparameters(n_trials=20)
```

---

## Version and Compatibility

- **Python**: ≥ 3.8
- **PyTorch**: ≥ 1.10.0
- **scvi-tools**: ≥ 0.17.0
- **XGBoost**: ≥ 1.5.0
- **scikit-learn**: ≥ 1.0.0

---

## Citation

If using CellLabeller in publications, please cite:

```bibtex
@software{celllabeller2024,
  title={CellLabeller: XGBoost-based cell type label transfer using scVI integration},
  year={2024},
  url={https://github.com/yourusername/CellLabeller}
}
```
