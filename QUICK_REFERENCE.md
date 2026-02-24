# CellLabeller Quick Reference

## Installation

```bash
git clone https://github.com/yourusername/CellLabeller.git
cd CellLabeller
pip install -r requirements.txt
pip install -e .
```

## Quick Start (5 Steps)

```python
from celllabeller import CellTypeLabelTransfer, FeatureEngineer, XGBoostTuner

# 1. Initialize
label_transfer = CellTypeLabelTransfer(
    reference_adata=adata_ref,
    query_adata=adata_query,
    n_epochs=250
)

# 2. Preprocess
label_transfer.subset_common_genes()
adata_integrated = label_transfer.integrate_with_scvi()

# 3. Features
fe = FeatureEngineer(adata_integrated)
X_features, _, _ = fe.select_features(feature_type="combined")

# 4. Train
tuner = XGBoostTuner(X_features, y_labels, label_transfer.get_results_dir())
comparison = tuner.compare_gpu_cpu()

# 5. Predict
y_pred = tuner.best_models["gpu_false"].predict(X_query)
```

## Common Parameters

| Parameter | Values | Default | Notes |
|-----------|--------|---------|-------|
| `n_epochs` | ≥200 | 250 | scVI training epochs |
| `feature_type` | "genes", "scvi_latent", "combined" | - | Feature selection |
| `n_features_genes` | 100-2000 | 500 | Number of top genes |
| `n_trials` | 10-200 | 50 | Hyperparameter trials |
| `use_gpu` | True/False | True | GPU acceleration |

## Key Methods

```python
# CellTypeLabelTransfer
label_transfer.subset_common_genes()           # → ref_adata, query_adata
label_transfer.integrate_with_scvi()           # → integrated_adata
label_transfer.get_results_dir()               # → results_path
label_transfer.save_integrated_data()          # Saves H5AD file

# FeatureEngineer
feature_engineer.select_features()             # → X_features, indices, names
feature_engineer.get_feature_importance_genes()  # → importance_df

# XGBoostTuner
tuner.tune_hyperparameters()                   # → best_params
tuner.train_best_model()                       # → trained_model
tuner.evaluate_model()                         # → results_dict
tuner.compare_gpu_cpu()                        # → comparison_df
tuner.save_model()                             # Saves .pkl files
tuner.save_results()                           # Saves metrics
```

## Output Files

```
results/
├── scvi_model/                 # scVI model directory
├── xgboost_model_cpu.pkl       # Trained CPU model
├── xgboost_model_gpu.pkl       # Trained GPU model
├── label_encoder.pkl           # Label encoder
├── evaluation_results_*.pkl    # Performance metrics
├── tuning_results_*.pkl        # Hyperparameter results
├── query_predictions.csv       # Predictions
├── query_with_predictions.h5ad # Query + predictions
└── *.png                        # Visualizations
```

## Feature Types Comparison

| Feature Type | Pros | Cons | Best For |
|--------------|------|------|----------|
| **genes** | Interpretable, stable | May miss integration patterns | Biological validation |
| **scvi_latent** | Captures integration, stable | Less interpretable | Integration-driven classification |
| **combined** | Best performance, comprehensive | More features, slower | Overall accuracy |

## Performance Expectations

| Step | GPU Time | CPU Time |
|------|----------|----------|
| Subsetting | <1s | <1s |
| scVI (250 epochs) | 5-15 min | 30-60 min |
| Features | <1 min | <1 min |
| Hyperparameter (50 trials) | 5-20 min | 30-60 min |
| Training | <1 min | <1 min |
| Prediction | <1s | <1s |

## Troubleshooting

### GPU not available
```python
label_transfer = CellTypeLabelTransfer(..., device="cpu")
# Or
tuner.tune_hyperparameters(use_gpu=False)
```

### Memory issues
```python
# Use fewer genes
feature_engineer = FeatureEngineer(..., n_features_genes=300)

# Reduce scVI epochs
label_transfer = CellTypeLabelTransfer(..., n_epochs=200)

# Fewer hyperparameter trials
tuner.tune_hyperparameters(n_trials=20)
```

### scVI not converging
```python
# Increase epochs
label_transfer = CellTypeLabelTransfer(..., n_epochs=500)
```

## Loading Saved Models

```python
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model and encoder
with open("results/xgboost_model_cpu.pkl", "rb") as f:
    model = pickle.load(f)

with open("results/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Predict
y_pred = encoder.inverse_transform(model.predict(new_features))
```

## Evaluation Metrics

- **Accuracy**: Overall proportion of correct predictions
- **Balanced Accuracy**: Average recall for each class (better for imbalanced data)
- **F1-Score (weighted)**: Harmonic mean of precision and recall, weighted by class frequency
- **Confusion Matrix**: Cell type classification breakdown

## Cell Type Quality Checks

```python
# Check predictions confidence
confidence = y_pred_proba.max(axis=1)
uncertain_cells = np.where(confidence < 0.7)[0]
print(f"Cells with <70% confidence: {len(uncertain_cells)}")

# Check per-class confidence
for i, ct in enumerate(model.classes_):
    mask = y_pred == i
    if mask.sum() > 0:
        avg_conf = y_pred_proba[mask, i].mean()
        print(f"{ct}: {avg_conf:.3f}")
```

## Recommended Workflows

### Exploration (Fast)
```
n_epochs=200, n_trials=20, feature_type="combined"
Time: ~30 min (GPU)
```

### Standard (Balanced)
```
n_epochs=250, n_trials=50, feature_type="combined"
Time: ~1-2 hours (GPU)
```

### Publication (Thorough)
```
n_epochs=500, n_trials=100, feature_type="combined" + cross-validation
Time: ~4-6 hours (GPU)
```

## Citation

```bibtex
@software{celllabeller2024,
  title={CellLabeller: XGBoost-based cell type label transfer using scVI integration},
  year={2024},
  url={https://github.com/yourusername/CellLabeller}
}
```

## Documentation Links

- **README**: Full overview and installation
- **API_DOCUMENTATION.md**: Complete API reference
- **tutorial_label_transfer.ipynb**: Interactive tutorial
- **DEVELOPMENT.md**: Contributing guide
- **examples/basic_workflow.py**: Runnable example

## Support

- 📖 Check the tutorial notebook
- 📚 Read the API documentation
- 🐛 Report issues on GitHub
- 💬 Discussion forum (coming soon)

---

**Last Updated**: February 2024
**Version**: 0.1.0
