#!/usr/bin/env python
"""
Memory diagnostic tool for cell type label transfer.
Analyzes dataset sizes and predicts memory usage.
"""

import os
import sys
import gc
import numpy as np
import anndata as ad
import scanpy as sc
from pathlib import Path
from scipy.sparse import issparse, csr_matrix
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_memory_gb(n_elements: int, dtype_size: int = 4) -> float:
    """Estimate memory in GB for sparse matrix."""
    # Sparse CSR: values + row indices + column pointers
    # Roughly: n_values * dtype_size + n_values * 4 + (n_rows+1) * 4
    # For typical sparsity (~95%), this is ~0.5 * n_values * dtype_size
    return (n_elements * dtype_size) / 1e9


def analyze_dataset(path: str, name: str = "Dataset") -> dict:
    """Analyze a single dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing {name}: {path}")
    logger.info(f"{'='*60}")
    
    if not Path(path).exists():
        logger.error(f"File not found: {path}")
        return None
    
    # Check file size
    file_size_gb = Path(path).stat().st_size / 1e9
    logger.info(f"File size: {file_size_gb:.2f} GB")
    
    # Load metadata without loading full data
    adata = ad.read_h5ad(path, backed='r')
    
    n_cells = adata.shape[0]
    n_genes = adata.shape[1]
    
    logger.info(f"Shape: {n_cells:,} cells × {n_genes:,} genes")
    logger.info(f"Sparsity: {(1 - (adata.X.nnz / (n_cells * n_genes))) * 100:.1f}%")
    logger.info(f"Data type: {adata.X.dtype}")
    logger.info(f"Sparse format: {type(adata.X).__name__}")
    
    if hasattr(adata, 'obs') and 'cell_type' in adata.obs.columns:
        ct_counts = adata.obs['cell_type'].value_counts()
        logger.info(f"Cell types: {len(ct_counts)}")
        logger.info(f"  Most abundant: {ct_counts.index[0]} ({ct_counts.iloc[0]:,} cells)")
        logger.info(f"  Least abundant: {ct_counts.index[-1]} ({ct_counts.iloc[-1]} cells)")
    
    # Memory estimate
    dense_gb = get_memory_gb(n_cells * n_genes, 4)
    sparse_gb = dense_gb * (1 - (1 - adata.X.nnz / (n_cells * n_genes)))
    
    logger.info(f"\nMemory estimates (float32):")
    logger.info(f"  If dense: {dense_gb:.2f} GB")
    logger.info(f"  If sparse: {sparse_gb:.2f} GB")
    logger.info(f"  With 3x peak multiplier: {sparse_gb * 3:.2f} GB")
    
    adata.file.close()
    del adata
    gc.collect()
    
    return {
        'name': name,
        'path': path,
        'n_cells': n_cells,
        'n_genes': n_genes,
        'file_size_gb': file_size_gb,
        'sparse_gb': sparse_gb,
        'peak_gb': sparse_gb * 3,
    }


def compare_concat_strategies(ref_info: dict, query_info: dict) -> None:
    """Compare memory usage for different concatenation strategies."""
    logger.info(f"\n{'='*60}")
    logger.info("Concatenation Strategy Comparison")
    logger.info(f"{'='*60}\n")
    
    ref_cells = ref_info['n_cells']
    ref_genes = ref_info['n_genes']
    query_cells = query_info['n_cells']
    query_genes = query_info['n_genes']
    
    # Estimate gene overlap (conservative: 80% overlap)
    overlap_fraction = 0.8
    common_genes = int(min(ref_genes, query_genes) * overlap_fraction)
    union_genes = int(ref_genes + query_genes - common_genes)
    
    total_cells = ref_cells + query_cells
    
    logger.info(f"Reference: {ref_cells:,} cells × {ref_genes:,} genes")
    logger.info(f"Query: {query_cells:,} cells × {query_genes:,} genes")
    logger.info(f"Estimated common genes: {common_genes:,} ({overlap_fraction*100:.0f}%)")
    logger.info(f"Estimated union genes: {union_genes:,}")
    logger.info(f"Total cells: {total_cells:,}\n")
    
    # Outer join (union)
    outer_elements = total_cells * union_genes
    outer_gb = get_memory_gb(outer_elements, 4)
    logger.info(f"Outer join (union of genes):")
    logger.info(f"  Dimensions: {total_cells:,} × {union_genes:,}")
    logger.info(f"  Memory: {outer_gb:.2f} GB (sparse)")
    logger.info(f"  Peak with 3x multiplier: {outer_gb * 3:.2f} GB")
    logger.info(f"  ❌ LIKELY TO CAUSE OOM\n")
    
    # Inner join (intersection)
    inner_elements = total_cells * common_genes
    inner_gb = get_memory_gb(inner_elements, 4)
    logger.info(f"Inner join (common genes only):")
    logger.info(f"  Dimensions: {total_cells:,} × {common_genes:,}")
    logger.info(f"  Memory: {inner_gb:.2f} GB (sparse)")
    logger.info(f"  Peak with 3x multiplier: {inner_gb * 3:.2f} GB")
    logger.info(f"  ✅ RECOMMENDED\n")
    
    savings = ((outer_gb - inner_gb) / outer_gb) * 100
    logger.info(f"Memory savings with inner join: {savings:.1f}%")
    logger.info(f"Absolute savings: {(outer_gb - inner_gb):.2f} GB")


def main():
    """Main diagnostic function."""
    logger.info("\n" + "="*60)
    logger.info("CellLabeller Memory Diagnostic Tool")
    logger.info("="*60)
    
    # Check if running from correct directory
    if not Path("git_repo/CellLabeller").exists():
        logger.warning("Script not in correct directory, attempting to find data...")
    
    # Paths (adjust as needed)
    ref_path = "/home/woody/mfn3/mfn3100h/aging/adata_v5_inguinal.h5ad"
    query_path = "/home/woody/mfn3/mfn3100h/region_AB11049/adata_processed.h5ad"
    
    # Analyze datasets
    ref_info = analyze_dataset(ref_path, "Reference (aging/adata_v5_inguinal)")
    if ref_info is None:
        logger.error("Failed to load reference dataset")
        return
    
    query_info = analyze_dataset(query_path, "Query (region_AB11049)")
    if query_info is None:
        logger.error("Failed to load query dataset")
        return
    
    # Compare strategies
    compare_concat_strategies(ref_info, query_info)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Summary and Recommendations")
    logger.info(f"{'='*60}\n")
    
    total_peak = max(ref_info['peak_gb'], query_info['peak_gb'])
    logger.info(f"Estimated peak memory (loading both sequentially): {total_peak:.2f} GB")
    logger.info(f"Available memory: 300 GB (recommended from 325GB)")
    logger.info(f"Safety margin: {300 - total_peak:.2f} GB\n")
    
    if total_peak < 300:
        logger.info("✅ GOOD: Peak memory within safe limits")
        logger.info("✅ Recommended: Use inner join (default in v0.1.1)")
        logger.info("✅ Expected success rate: >95%")
    elif total_peak < 320:
        logger.info("⚠️  CAUTION: Peak memory close to limit")
        logger.info("⚠️  Recommendations:")
        logger.info("   1. Use inner join (implemented in v0.1.1)")
        logger.info("   2. Filter reference by cell type (keep abundant types only)")
        logger.info("   3. Use float32 (already in your script)")
        logger.info("   4. Monitor with: sstat -j <JobID> --format=MaxRSS")
    else:
        logger.info("❌ CRITICAL: Peak memory likely exceeds available RAM")
        logger.info("❌ Recommendations:")
        logger.info("   1. Subsample reference to ~80k cells")
        logger.info("   2. Filter to top cell types only")
        logger.info("   3. Use HDF5 backed mode")
        logger.info("   4. Consider splitting analysis by batch")
    
    logger.info(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
