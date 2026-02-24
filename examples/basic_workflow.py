"""
Basic example: Cell type label transfer workflow

This example demonstrates the complete CellLabeller workflow:
1. Load reference and query datasets
2. Subset to common genes
3. Integrate with scVI
4. Select features
5. Tune hyperparameters
6. Train and evaluate model
7. Make predictions on query cells

Requirements:
- Reference AnnData object with cell_type in .obs
- Query AnnData object
"""

import numpy as np
import anndata as ad
from pathlib import Path
from celllabeller import CellTypeLabelTransfer, FeatureEngineer, XGBoostTuner


def main():
    """Run basic label transfer workflow."""
    
    # ========================================
    # 1. LOAD DATA
    # ========================================
    print("Step 1: Loading data...")
    
    # Replace these paths with your actual data
    ref_path = "path/to/reference_data.h5ad"
    query_path = "path/to/query_data.h5ad"
    
    try:
        adata_ref = ad.read_h5ad(ref_path)
        adata_query = ad.read_h5ad(query_path)
    except FileNotFoundError:
        print(f"Error: Could not find data files at:")
        print(f"  Reference: {ref_path}")
        print(f"  Query: {query_path}")
        print("Please update the paths and try again.")
        return
    
    print(f"  Reference shape: {adata_ref.shape}")
    print(f"  Query shape: {adata_query.shape}")
    
    # ========================================
    # 2. INITIALIZE LABEL TRANSFER
    # ========================================
    print("\nStep 2: Initializing label transfer...")
    
    results_dir = "./celllabeller_results"
    cell_type_key = "cell_type"  # Update if your column name is different
    
    label_transfer = CellTypeLabelTransfer(
        reference_adata=adata_ref,
        query_adata=adata_query,
        cell_type_key=cell_type_key,
        results_dir=results_dir,
        n_epochs=250,  # At least 200 as required
        device="gpu",  # Set to "cpu" if GPU not available
    )
    
    # ========================================
    # 3. SUBSET TO COMMON GENES
    # ========================================
    print("\nStep 3: Subsetting to common genes...")
    
    adata_ref, adata_query = label_transfer.subset_common_genes()
    print(f"  Common genes: {adata_ref.shape[1]}")
    
    # ========================================
    # 4. INTEGRATE WITH scVI
    # ========================================
    print("\nStep 4: Integrating with scVI...")
    print("  (This may take several minutes...)")
    
    adata_integrated = label_transfer.integrate_with_scvi()
    print(f"  Integrated shape: {adata_integrated.shape}")
    print(f"  Latent space: {adata_integrated.obsm['X_scvi'].shape}")
    
    # ========================================
    # 5. FEATURE ENGINEERING
    # ========================================
    print("\nStep 5: Feature engineering...")
    
    feature_engineer = FeatureEngineer(
        integrated_adata=adata_integrated,
        cell_type_key=cell_type_key,
        n_features_genes=500,
    )
    
    # Choose feature type: "genes", "scvi_latent", or "combined"
    feature_type = "combined"
    
    X_features, indices, feature_names = feature_engineer.select_features(
        feature_type=feature_type
    )
    
    print(f"  Feature type: {feature_type}")
    print(f"  Total features: {X_features.shape[1]}")
    
    # ========================================
    # 6. HYPERPARAMETER TUNING
    # ========================================
    print("\nStep 6: Hyperparameter tuning...")
    
    tuner = XGBoostTuner(
        X_features=X_features,
        y_labels=adata_integrated.obs[cell_type_key].values,
        results_dir=label_transfer.get_results_dir(),
    )
    
    # Option 1: Tune both GPU and CPU (recommended for comparison)
    print("  Running GPU vs CPU comparison...")
    comparison_df = tuner.compare_gpu_cpu()
    
    print("\nPerformance Comparison:")
    print(comparison_df)
    
    # ========================================
    # 7. MAKE PREDICTIONS ON QUERY
    # ========================================
    print("\nStep 7: Making predictions on query cells...")
    
    query_mask = adata_integrated.obs["dataset"] == "query"
    X_query = X_features[query_mask]
    
    # Use the best CPU model (always available)
    best_model = tuner.best_models.get("gpu_false")
    if best_model is None:
        print("Error: No model available for prediction")
        return
    
    y_query_pred = best_model.predict(X_query)
    y_query_proba = best_model.predict_proba(X_query)
    
    # Decode predictions
    y_query_labels = tuner.label_encoder.inverse_transform(y_query_pred)
    
    # ========================================
    # 8. DISPLAY RESULTS
    # ========================================
    print("\nStep 8: Results")
    print("=" * 60)
    
    print(f"\nQuery cells: {len(y_query_labels)}")
    print("\nPredicted cell type distribution:")
    unique, counts = np.unique(y_query_labels, return_counts=True)
    for ct, count in zip(unique, counts):
        pct = (count / len(y_query_labels)) * 100
        print(f"  {ct}: {count} ({pct:.1f}%)")
    
    print(f"\nMean prediction confidence: {y_query_proba.max(axis=1).mean():.4f}")
    print(f"Min confidence: {y_query_proba.max(axis=1).min():.4f}")
    print(f"Max confidence: {y_query_proba.max(axis=1).max():.4f}")
    
    # ========================================
    # 9. SAVE PREDICTIONS
    # ========================================
    print("\nStep 9: Saving predictions...")
    
    import pandas as pd
    
    # Add predictions to query AnnData
    adata_query.obs["predicted_cell_type"] = y_query_labels
    adata_query.obs["prediction_confidence"] = y_query_proba.max(axis=1)
    
    # Save query with predictions
    results_path = Path(results_dir)
    query_pred_path = results_path / "query_with_predictions.h5ad"
    adata_query.write_h5ad(str(query_pred_path))
    print(f"  Saved: {query_pred_path}")
    
    # Save predictions as CSV
    pred_df = pd.DataFrame({
        "cell_id": adata_query.obs_names,
        "predicted_cell_type": y_query_labels,
        "confidence": y_query_proba.max(axis=1),
    })
    
    csv_path = results_path / "predictions.csv"
    pred_df.to_csv(str(csv_path), index=False)
    print(f"  Saved: {csv_path}")
    
    print(f"\n✅ Analysis complete! Results saved to: {results_path}")


if __name__ == "__main__":
    main()
