"""
Feature engineering module for selecting features from genes and/or scVI latent space
"""

import numpy as np
import pandas as pd
import anndata as ad
from typing import Tuple, Literal, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for preparing features from both gene expression
    and scVI latent space representations.
    
    Parameters
    ----------
    integrated_adata : anndata.AnnData
        Integrated dataset with scVI latent representation
    cell_type_key : str, default "cell_type"
        Column name containing cell type labels
    n_features_genes : int, default 500
        Number of top genes to select
    """
    
    def __init__(
        self,
        integrated_adata: ad.AnnData,
        cell_type_key: str = "cell_type",
        n_features_genes: int = 500,
    ):
        self.integrated_adata = integrated_adata.copy()
        self.cell_type_key = cell_type_key
        self.n_features_genes = n_features_genes
        
        self.X_features = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized FeatureEngineer with {n_features_genes} genes")
    
    def select_features(
        self,
        feature_type: Literal["genes", "scvi_latent", "combined"] = "combined",
        n_scvi_components: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Select features based on user preference.
        
        Parameters
        ----------
        feature_type : {"genes", "scvi_latent", "combined"}, default "combined"
            Type of features to use:
            - "genes": Use top selected genes based on feature importance
            - "scvi_latent": Use scVI latent space representation
            - "combined": Concatenate both genes and scVI latent space
        n_scvi_components : int, optional
            Number of scVI components to use. If None, uses all available
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, list]
            - Feature matrix (n_cells x n_features)
            - Corresponding cell indices for reference and query
            - Feature names/identifiers
        """
        logger.info(f"Selecting features with type: {feature_type}")
        
        if feature_type == "genes":
            X, indices, feature_names = self._select_gene_features()
        elif feature_type == "scvi_latent":
            X, indices, feature_names = self._select_scvi_features(n_scvi_components)
        elif feature_type == "combined":
            X_genes, idx_genes, names_genes = self._select_gene_features()
            X_scvi, idx_scvi, names_scvi = self._select_scvi_features(n_scvi_components)
            
            # Both should have same indices
            assert np.array_equal(idx_genes, idx_scvi), "Indices mismatch"
            
            X = np.hstack([X_genes, X_scvi])
            indices = idx_genes
            feature_names = names_genes + names_scvi
            
            logger.info(f"Combined features: {len(names_genes)} genes + {len(names_scvi)} scVI components")
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        
        # Normalize features
        logger.info("Standardizing features...")
        X = self.scaler.fit_transform(X)
        
        self.X_features = X
        self.feature_names = feature_names
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        
        return X, indices, feature_names
    
    def _select_gene_features(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """Select top genes based on f_classif."""
        logger.info("Selecting top genes...")
        
        # Get reference data only
        ref_mask = self.integrated_adata.obs["dataset"] == "reference"
        X_ref = self.integrated_adata.X[ref_mask]
        y_ref = self.integrated_adata.obs.loc[ref_mask, self.cell_type_key].values
        
        # Feature selection using f_classif
        selector = SelectKBest(f_classif, k=self.n_features_genes)
        X_selected_ref = selector.fit_transform(X_ref, y_ref)
        
        # Get selected gene indices
        selected_idx = selector.get_support(indices=True)
        selected_genes = self.integrated_adata.var_names[selected_idx].tolist()
        
        logger.info(f"Selected {len(selected_genes)} top genes")
        
        # Apply to full dataset
        X_selected = self.integrated_adata[:, selected_genes].X.toarray() if hasattr(
            self.integrated_adata[:, selected_genes].X, 'toarray'
        ) else self.integrated_adata[:, selected_genes].X
        
        indices = np.arange(self.integrated_adata.n_obs)
        
        return X_selected, indices, selected_genes
    
    def _select_scvi_features(
        self, n_components: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Select scVI latent features."""
        logger.info("Selecting scVI latent features...")
        
        if "X_scvi" not in self.integrated_adata.obsm:
            raise ValueError("X_scvi not found in obsm. Run integrate_with_scvi first.")
        
        X_scvi = self.integrated_adata.obsm["X_scvi"]
        
        if n_components is not None and n_components < X_scvi.shape[1]:
            X_scvi = X_scvi[:, :n_components]
            logger.info(f"Using {n_components} out of {self.integrated_adata.obsm['X_scvi'].shape[1]} scVI components")
        else:
            logger.info(f"Using all {X_scvi.shape[1]} scVI components")
        
        indices = np.arange(self.integrated_adata.n_obs)
        feature_names = [f"scvi_{i}" for i in range(X_scvi.shape[1])]
        
        return X_scvi, indices, feature_names
    
    def get_feature_importance_genes(self) -> pd.DataFrame:
        """
        Calculate feature importance for selected genes.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with genes and their importance scores
        """
        if self.feature_names is None or "scvi_" in self.feature_names[0]:
            raise ValueError("Cannot compute gene importance without selected genes")
        
        ref_mask = self.integrated_adata.obs["dataset"] == "reference"
        y_ref = self.integrated_adata.obs.loc[ref_mask, self.cell_type_key].values
        
        # Calculate F-scores for selected genes
        from sklearn.feature_selection import f_classif
        scores, _ = f_classif(self.X_features[ref_mask], y_ref)
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": scores,
        }).sort_values("importance", ascending=False)
        
        return importance_df
