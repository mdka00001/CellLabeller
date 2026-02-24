"""
Core module for integrating reference and query datasets using scVI
"""

import os
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvi
from typing import Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CellTypeLabelTransfer:
    """
    A class to perform cell type label transfer from reference to query datasets
    using scVI integration and XGBoost classification.
    
    Parameters
    ----------
    reference_adata : anndata.AnnData
        Reference single-cell dataset with cell type annotations
    query_adata : anndata.AnnData
        Query dataset to be labeled
    cell_type_key : str, default "cell_type"
        Column name in reference_adata.obs containing cell type labels
    batch_key : str, optional
        Column name for batch correction in scVI
    results_dir : str, default "./celllabeller_results"
        Directory to store results and models
    n_epochs : int, default 200
        Number of epochs for scVI training
    device : str, default "gpu"
        Device to use: "gpu" or "cpu"
    """
    
    def __init__(
        self,
        reference_adata: ad.AnnData,
        query_adata: ad.AnnData,
        cell_type_key: str = "cell_type",
        batch_key: Optional[str] = None,
        results_dir: str = "./celllabeller_results",
        n_epochs: int = 200,
        device: str = "gpu",
    ):
        self.reference_adata = reference_adata.copy()
        self.query_adata = query_adata.copy()
        self.cell_type_key = cell_type_key
        self.batch_key = batch_key
        self.n_epochs = max(n_epochs, 200)  # Ensure at least 200 epochs
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.integrated_adata = None
        self.feature_engineer = None
        self.tuner = None
        
        logger.info(f"Initialized CellTypeLabelTransfer with {n_epochs} epochs")
        logger.info(f"Reference shape: {reference_adata.shape}, Query shape: {query_adata.shape}")
    
    def subset_common_genes(self) -> Tuple[ad.AnnData, ad.AnnData]:
        """
        Subset both reference and query datasets to common genes.
        
        Returns
        -------
        Tuple[anndata.AnnData, anndata.AnnData]
            Reference and query AnnData objects with common genes only
        """
        logger.info("Subsetting to common genes...")
        
        ref_genes = set(self.reference_adata.var_names)
        query_genes = set(self.query_adata.var_names)
        common_genes = sorted(list(ref_genes.intersection(query_genes)))
        
        logger.info(f"Common genes: {len(common_genes)}")
        logger.info(f"Reference genes removed: {len(ref_genes) - len(common_genes)}")
        logger.info(f"Query genes removed: {len(query_genes) - len(common_genes)}")
        
        self.reference_adata = self.reference_adata[:, common_genes]
        self.query_adata = self.query_adata[:, common_genes]
        
        return self.reference_adata, self.query_adata
    
    def integrate_with_scvi(self) -> ad.AnnData:
        """
        Integrate reference and query datasets using scVI.
        
        Returns
        -------
        anndata.AnnData
            Integrated dataset with scVI latent representation
        """
        logger.info("Preparing data for scVI integration...")
        
        # IMPORTANT: Subset to common genes FIRST to reduce memory usage
        logger.info(f"Reference shape before subset: {self.reference_adata.shape}")
        logger.info(f"Query shape before subset: {self.query_adata.shape}")
        
        if len(self.reference_adata.var) != len(self.query_adata.var):
            logger.warning("Reference and Query have different genes. Subsetting to common genes...")
            self.subset_common_genes()
        
        logger.info(f"Reference shape after subset: {self.reference_adata.shape}")
        logger.info(f"Query shape after subset: {self.query_adata.shape}")
        
        # Add dataset identifier
        self.reference_adata.obs["dataset"] = "reference"
        self.query_adata.obs["dataset"] = "query"
        
        # Concatenate datasets with inner join (only common genes, already subset above)
        logger.info("Concatenating datasets with inner join (memory-efficient)...")
        integrated = sc.concat(
            [self.reference_adata, self.query_adata],
            join="inner",  # Changed from "outer" to "inner" for memory efficiency
            label="dataset_id",
            keys=["reference", "query"],
            axis=0,
        )
        
        logger.info(f"Integrated dataset shape: {integrated.shape}")
        logger.info(f"Integrated matrix type: {type(integrated.X)}, dtype: {integrated.X.dtype}")
        
        # Ensure sparse matrix representation (memory efficient)
        if not hasattr(integrated.X, 'toarray'):
            logger.warning("Converting to sparse matrix...")
            from scipy.sparse import csr_matrix
            integrated.X = csr_matrix(integrated.X)
        
        # Normalize and log transform (keep sparse)
        logger.info("Preprocessing data (preserving sparse matrix)...")
        sc.pp.normalize_total(integrated, target_sum=1e4)
        sc.pp.log1p(integrated)
        
        # Setup scVI with batch correction on dataset_id
        logger.info("Setting up scVI model...")
        scvi.model.SCVI.setup_anndata(
            integrated,
            batch_key="dataset_id",  # Use the key from concat
            batch_correction=True,
        )
        
        # Train scVI model with memory optimization
        logger.info(f"Training scVI model for {self.n_epochs} epochs...")
        model = scvi.model.SCVI(integrated, n_latent=30)  # 30 latent dims sufficient
        model.train(
            max_epochs=self.n_epochs,
            early_stopping=True,
            early_stopping_patience=10,
            accelerator="gpu" if self.device == "gpu" else "cpu",
        )
        
        # Get latent representation
        logger.info("Extracting scVI latent representation...")
        integrated.obsm["X_scvi"] = model.get_latent_representation()
        
        # Save model
        model_path = self.results_dir / "scvi_model"
        logger.info(f"Saving scVI model to {model_path}")
        model.save(str(model_path), overwrite=True)
        
        # Clear model from memory
        del model
        import gc
        gc.collect()
        
        self.integrated_adata = integrated
        
        return integrated
    
    def get_results_dir(self) -> Path:
        """Get the results directory path."""
        return self.results_dir
    
    def get_integrated_adata(self) -> ad.AnnData:
        """Get the integrated AnnData object."""
        if self.integrated_adata is None:
            raise ValueError("Data must be integrated first using integrate_with_scvi()")
        return self.integrated_adata
    
    def save_integrated_data(self, filename: str = "integrated_data.h5ad"):
        """
        Save integrated data to disk.
        
        Parameters
        ----------
        filename : str
            Filename for saving
        """
        if self.integrated_adata is None:
            raise ValueError("No integrated data available")
        
        save_path = self.results_dir / filename
        logger.info(f"Saving integrated data to {save_path}")
        self.integrated_adata.write_h5ad(str(save_path))
