# Getting Started with CellLabeller

Welcome to CellLabeller! This guide will help you get up and running in 5 minutes.

## What is CellLabeller?

CellLabeller is a tool that automatically transfers cell type labels from a reference single-cell dataset to a query dataset using:
- **scVI**: For integrating datasets
- **XGBoost**: For classification
- **Optuna**: For hyperparameter optimization

It works with your single-cell RNA-seq data in the popular AnnData format.

## Installation (2 minutes)

### Step 1: Clone the Repository

```bash
cd /path/to/your/workspace
git clone https://github.com/yourusername/CellLabeller.git
cd CellLabeller
```

### Step 2: Install Dependencies

```bash
# Option A: Using pip (recommended)
pip install -r requirements.txt
pip install -e .

# Option B: Using conda (if you have conda)
conda create -n celllabeller python=3.9
conda activate celllabeller
pip install -r requirements.txt
pip install -e .
```

### Step 3: Verify Installation

```python
python -c "import celllabeller; print('✓ Installation successful!')"
```

## Prepare Your Data (2 minutes)

You need two AnnData objects in H5AD format:

### Reference Data Requirements
```python
import anndata as ad

adata_ref = ad.read_h5ad("reference_data.h5ad")

# Must have cell type information
assert "cell_type" in adata_ref.obs, "Reference must have cell_type column"

# Check structure
print(adata_ref)
# Output: AnnData object with n_obs × n_vars = 50000 × 20000
#         obs: 'cell_type', ...
```

### Query Data Requirements
```python
adata_query = ad.read_h5ad("query_data.h5ad")

# Can be unlabeled (labels will be predicted)
print(adata_query)
# Output: AnnData object with n_obs × n_vars = 10000 × 20000
```

## Run Your First Analysis (1 minute)

### Complete Pipeline

```python
from celllabeller import CellTypeLabelTransfer, FeatureEngineer, XGBoostTuner
import anndata as ad

# Load your data
adata_ref = ad.read_h5ad("reference_data.h5ad")
adata_query = ad.read_h5ad("query_data.h5ad")

# 1. Initialize
print("Step 1: Initializing...")
label_transfer = CellTypeLabelTransfer(
    reference_adata=adata_ref,
    query_adata=adata_query,
    cell_type_key="cell_type",
    results_dir="./my_results",
    n_epochs=250,  # At least 200
)

# 2. Subset and integrate
print("Step 2: Subsetting and integrating...")
label_transfer.subset_common_genes()
adata_integrated = label_transfer.integrate_with_scvi()

# 3. Select features
print("Step 3: Selecting features...")
fe = FeatureEngineer(adata_integrated, cell_type_key="cell_type")
X_features, _, _ = fe.select_features(feature_type="combined")

# 4. Train model
print("Step 4: Training model (this may take 10-30 minutes)...")
tuner = XGBoostTuner(
    X_features=X_features,
    y_labels=adata_integrated.obs["cell_type"].values,
    results_dir=label_transfer.get_results_dir(),
)

# 5. Make predictions
print("Step 5: Making predictions...")
query_mask = adata_integrated.obs["dataset"] == "query"
X_query = X_features[query_mask]
y_pred = tuner.best_models["gpu_false"].predict(X_query)
y_proba = tuner.best_models["gpu_false"].predict_proba(X_query)

print("✓ Done!")
print(f"Predicted {len(y_pred)} query cells")
print(f"Mean confidence: {y_proba.max(axis=1).mean():.2%}")
```

## Where's My Results?

All results are saved in the `results_dir` (default: `./celllabeller_results/`):

```
celllabeller_results/
├── query_predictions.csv          ← Cell type predictions (open in Excel!)
├── query_with_predictions.h5ad    ← AnnData with predictions
├── xgboost_model_cpu.pkl          ← Trained model
├── label_encoder.pkl              ← Label encoder
├── evaluation_results_cpu.pkl      ← Performance metrics
├── confusion_matrices.png          ← Visualizations
└── ... (other files for reproducibility)
```

**Quick Look at Predictions**:
```python
import pandas as pd

# Load predictions as DataFrame
pred_df = pd.read_csv("celllabeller_results/query_predictions.csv")
print(pred_df.head())
#                  cell_id predicted_cell_type  confidence
# 0  cell_1234         T_cell         0.95
# 1  cell_5678         B_cell         0.87
# ...

# Or load the AnnData with predictions
import anndata as ad
adata_pred = ad.read_h5ad("celllabeller_results/query_with_predictions.h5ad")
print(adata_pred.obs[["predicted_cell_type", "prediction_confidence"]].head())
```

## Customization

### Feature Types
Choose what features to use for classification:

```python
# Option 1: Gene expression only (most interpretable)
X, _, _ = fe.select_features(feature_type="genes")

# Option 2: scVI latent space only (best for integration)
X, _, _ = fe.select_features(feature_type="scvi_latent")

# Option 3: Combined (best overall accuracy - recommended!)
X, _, _ = fe.select_features(feature_type="combined")
```

### Number of Genes
```python
fe = FeatureEngineer(adata_integrated, n_features_genes=1000)  # Use more genes
```

### GPU/CPU
```python
# If you have a GPU:
label_transfer = CellTypeLabelTransfer(..., device="gpu")

# If you only have CPU:
label_transfer = CellTypeLabelTransfer(..., device="cpu")
```

### Hyperparameter Tuning
```python
# Quick (30 minutes)
best_params = tuner.tune_hyperparameters(n_trials=20)

# Standard (1-2 hours)
best_params = tuner.tune_hyperparameters(n_trials=50)

# Thorough (4-6 hours)
best_params = tuner.tune_hyperparameters(n_trials=100)
```

## Troubleshooting

### "GPU not available"
```python
# Use CPU instead
label_transfer = CellTypeLabelTransfer(..., device="cpu")
```

### "Out of memory"
```python
# Use fewer genes
fe = FeatureEngineer(adata_integrated, n_features_genes=300)

# Or reduce training
label_transfer = CellTypeLabelTransfer(..., n_epochs=200)
```

### "slow integration"
This is normal! scVI integration takes time. 
- GPU: 5-15 minutes for 50k cells
- CPU: 30-60 minutes for 50k cells

### "Poor predictions"
Check:
1. Is your reference data high quality?
2. Are your cell types well-defined?
3. Do reference and query come from similar experiments?
4. Try: combine feature types, increase genes, increase hyperparameter trials

## Next Steps

1. **Check the tutorial**: Run [tutorial_label_transfer.ipynb](tutorial_label_transfer.ipynb)
2. **Read the README**: See [README.md](README.md) for more details
3. **API reference**: Check [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
4. **Quick reference**: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

## Example: Complete Workflow

Here's a minimal working example you can copy:

```python
#!/usr/bin/env python
"""Minimal CellLabeller example."""

import anndata as ad
from celllabeller import CellTypeLabelTransfer, FeatureEngineer, XGBoostTuner

# ============================================
# CUSTOMIZE THESE PATHS
# ============================================
REF_PATH = "reference.h5ad"
QUERY_PATH = "query.h5ad"
RESULTS_DIR = "./my_results"

# ============================================
# LOAD DATA
# ============================================
adata_ref = ad.read_h5ad(REF_PATH)
adata_query = ad.read_h5ad(QUERY_PATH)

print(f"Reference: {adata_ref.shape}")
print(f"Query: {adata_query.shape}")

# ============================================
# PIPELINE
# ============================================
label_transfer = CellTypeLabelTransfer(
    reference_adata=adata_ref,
    query_adata=adata_query,
    results_dir=RESULTS_DIR,
    n_epochs=250,
)

label_transfer.subset_common_genes()
adata_integrated = label_transfer.integrate_with_scvi()

fe = FeatureEngineer(adata_integrated)
X_features, _, _ = fe.select_features(feature_type="combined")

tuner = XGBoostTuner(
    X_features=X_features,
    y_labels=adata_integrated.obs["cell_type"].values,
    results_dir=label_transfer.get_results_dir(),
)

tuner.compare_gpu_cpu()

# ============================================
# PREDICT QUERY
# ============================================
query_mask = adata_integrated.obs["dataset"] == "query"
X_query = X_features[query_mask]
y_pred = tuner.best_models["gpu_false"].predict(X_query)

print(f"\n✓ Predicted {len(y_pred)} cells!")
```

## Getting Help

- 📖 **Documentation**: Check README.md and API_DOCUMENTATION.md
- 📓 **Tutorial**: Run the Jupyter notebook
- 🐛 **Issues**: Open an issue on GitHub
- 💡 **Ideas**: Check existing examples

## Performance Tips

| Goal | Strategy |
|------|----------|
| **Quick test** | Use 200 genes, 20 trials, CPU |
| **Good results** | Use 500-1000 genes, 50 trials |
| **Publication** | Use 1000 genes, 100+ trials, GPU |
| **Large dataset** | Reduce genes to 300, use GPU |

## Key Concepts

**scVI Integration**: Creates a shared latent space between reference and query datasets, correcting for batch effects and integrating information.

**Feature Selection**: You choose whether to use gene expression, scVI latent space, or both for training the classifier.

**XGBoost**: A powerful gradient boosting classifier that learns to predict cell types based on selected features.

**Hyperparameter Tuning**: Automatically finds the best settings for XGBoost using Bayesian optimization.

**GPU Acceleration**: Optional but recommended for faster processing on large datasets.

## Citation

If you use CellLabeller in your research, please cite:

```bibtex
@software{celllabeller2024,
  title={CellLabeller: XGBoost-based cell type label transfer using scVI integration},
  year={2024},
  url={https://github.com/yourusername/CellLabeller}
}
```

---

**You're all set!** 🎉

Start with the example code above, then check the tutorial notebook for more advanced usage.

Good luck with your cell type labeling!
