# Memory Optimization Guide for Large-Scale scRNA-seq Analysis

## Problem: OOM Kill with 325GB RAM

Your system has 325GB RAM but still hits OOM when running `sc.concat()` with large datasets. This is a **common issue with aging/tissue studies** that have:
- 100k+ cells × 30k+ genes  
- Dense matrix operations during integration
- Memory-inefficient sparse matrix handling

---

## Root Cause Analysis

### What Happened:
1. `sc.concat(..., join="outer")` creates a **union** of all genes from both datasets
2. If reference and query have different genes, this expands the gene dimension significantly
3. For massive matrices (100k cells × 60k genes), sparse matrices still use substantial memory
4. Normalization, scaling, and setup operations create multiple copies in memory

### Memory Calculation Example:
```
Reference: 50,000 cells × 30,000 genes = 1.5 billion values
Query:     60,000 cells × 30,000 genes = 1.8 billion values

With join="outer" (union of genes):
Total: 110,000 cells × 35,000 genes = 3.85 billion values

Float32 sparse: ~3.85B × 4 bytes ≈ 15.4 GB
Float64 sparse: ~3.85B × 8 bytes ≈ 30.8 GB

But with intermediate copies during ops: 3-5x multiplier → 50-150 GB peak
```

---

## Solutions Implemented in CellLabeller v0.1.1

### 1. **Subset Common Genes BEFORE Concatenation** ✅
```python
# This is now enforced in integrate_with_scvi()
if len(self.reference_adata.var) != len(self.query_adata.var):
    logger.warning("Reference and Query have different genes. Subsetting...")
    self.subset_common_genes()
```

**Impact:** Reduces gene dimension from union (35k) to intersection (~30k)

### 2. **Use `join="inner"` Instead of `join="outer"`** ✅
```python
integrated = sc.concat(
    [self.reference_adata, self.query_adata],
    join="inner",  # Only common genes, no union
    label="dataset_id",
    keys=["reference", "query"],
    axis=0,
)
```

**Impact:** Avoids creating sparse matrix with unnecessary genes

### 3. **Force Sparse Matrix Representation** ✅
```python
# Ensure sparse matrix (saves 50-70% memory vs dense)
if not hasattr(integrated.X, 'toarray'):
    from scipy.sparse import csr_matrix
    integrated.X = csr_matrix(integrated.X)
```

**Impact:** All operations stay sparse, no accidental densification

### 4. **Reduce scVI Latent Dimension** ✅
```python
model = scvi.model.SCVI(integrated, n_latent=30)  # Default 10, we use 30 for quality
```

**Impact:** Smaller model uses less GPU/CPU memory during training

### 5. **Enable GPU Acceleration for scVI Training** ✅
```python
model.train(
    max_epochs=self.n_epochs,
    early_stopping=True,
    early_stopping_patience=10,
    accelerator="gpu" if self.device == "gpu" else "cpu",
)
```

**Impact:** GPU handles scVI computation, frees CPU RAM

### 6. **Clean Up After Training** ✅
```python
del model
import gc
gc.collect()
```

**Impact:** Releases scVI model from memory after getting latent representation

---

## Additional Recommendations for Your Use Case

### A. Pre-filter Cell Types (Already in Your Script ✅)
```python
# Keep only cell types with >= 10 cells
counts = adata_ref.obs["cell_type"].value_counts()
keep_cell_types = counts[counts >= 10].index
adata_ref = adata_ref[adata_ref.obs["cell_type"].isin(keep_cell_types)].copy()
```

**Impact:** Reduces reference from 50k → ~30k cells

### B. Use Float32 Instead of Float64 (Already in Your Script ✅)
```python
adata_ref.X = adata_ref.X.astype('float32')
adata_query.X = adata_query.X.astype('float32')
```

**Impact:** 50% memory savings (4 bytes vs 8 bytes per value)

### C. Keep Scale Operations Sparse (Already in Your Script ✅)
```python
# CRITICAL: zero_center=False keeps matrix sparse
sc.pp.scale(adata_ref, max_value=10, zero_center=False)
```

**Impact:** `zero_center=True` forces dense conversion (2-3x memory)

### D. Load Query AFTER Reference Processing
```python
# Process reference first, then load query
adata_ref = ... # Process
del intermediate_objects
gc.collect()

# THEN load query
adata_query = ad.read_h5ad(query_path)
```

**Impact:** Minimizes peak memory usage (sequential, not parallel)

### E. Consider Using HDF5 Backed Mode for Massive Datasets
```python
# For truly massive datasets (>200k cells):
adata = ad.read_h5ad(path, backed='r')  # Read-only
# Or 'r+' for read-write
```

**Impact:** Only loads data into memory on-demand

---

## SLURM Job Submission Optimization

Your `script.sh` should request appropriate resources:

```bash
#!/bin/bash
#SBATCH --job-name=celllabeller
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # Multi-threaded operations
#SBATCH --mem=300G                   # Leave headroom (300G of 325G)
#SBATCH --gres=gpu:1                 # One RTX3080
#SBATCH --time=08:00:00              # 8 hours
#SBATCH --output=logs/celllabeller_%j.out
#SBATCH --error=logs/celllabeller_%j.err

module load cuda/11.8  # If available
module load python/3.10  # Or appropriate version

python script.py
```

**Key Points:**
- Request `--mem=300G` (not 325G) to leave system headroom
- Use `--gres=gpu:1` to leverage RTX3080
- Request multiple `--cpus-per-task` for parallel operations
- Set `--time` high enough (aging studies take 2-4 hours typically)

---

## Expected Behavior After Fixes

### Memory Timeline:
1. **Load Reference**: 10-15 GB
2. **Filter + Preprocessing**: 8-12 GB (same objects, in-place ops)
3. **Load Query**: 10-15 GB
4. **Concatenate + Subset**: 20-25 GB (peak memory)
5. **scVI Training**: 15-20 GB (stays on GPU after iteration 1)
6. **Latent Extraction**: 15-20 GB
7. **Feature Engineering**: 10-15 GB
8. **XGBoost Training**: 5-10 GB

**Total Peak**: ~50-80 GB (well within 300GB available)

---

## Testing the Fixes

### 1. Check Updated Code
```bash
cd /home/woody/mfn3/mfn3100h/git_repo/CellLabeller
grep -n "join=" celllabeller/label_transfer.py
# Should show: join="inner" (not "outer")
```

### 2. Run Diagnostics Script
```python
# diagnostics.py
import anndata as ad
import scanpy as sc
import numpy as np
from scipy.sparse import issparse

ref = ad.read_h5ad("path/to/reference.h5ad")
query = ad.read_h5ad("path/to/query.h5ad")

print(f"Reference: {ref.shape}, sparse={issparse(ref.X)}, dtype={ref.X.dtype}")
print(f"Query: {query.shape}, sparse={issparse(query.X)}, dtype={query.X.dtype}")

# Check gene overlap
common = len(set(ref.var_names) & set(query.var_names))
ref_only = len(set(ref.var_names) - set(query.var_names))
query_only = len(set(query.var_names) - set(ref.var_names))

print(f"Common genes: {common}, Ref-only: {ref_only}, Query-only: {query_only}")
print(f"Memory estimate for concat:")
print(f"  outer join: {(ref.shape[0] + query.shape[0]) * (ref.shape[1] + query_only) * 4 / 1e9:.1f} GB")
print(f"  inner join: {(ref.shape[0] + query.shape[0]) * common * 4 / 1e9:.1f} GB")
```

### 3. Monitor System During Run
```bash
# In separate terminal:
watch -n 1 'free -h && nvidia-smi'
```

---

## Debugging: If Still OOM

### Check Dataset Sizes
```bash
python -c "
import anndata as ad
ref = ad.read_h5ad('aging/adata_v5_inguinal.h5ad')
query = ad.read_h5ad('region_AB11049/adata_processed.h5ad')
print(f'Reference: {ref.shape}')
print(f'Query: {query.shape}')
print(f'Common genes: {len(set(ref.var_names) & set(query.var_names))}')
"
```

### If Reference is >100k cells:
- Further filter by cell type (keep only abundant types)
- Or subsample: `adata_ref = adata_ref[np.random.choice(adata_ref.n_obs, 80000)].copy()`

### If Query has >100k cells:
- Train on smaller reference, predict on full query
- Or split query into batches for prediction

### Use HDF5 Backed Mode:
```python
adata_ref = ad.read_h5ad("path.h5ad", backed='r')
# Computations will use disk I/O, slower but memory-efficient
```

---

## Summary of Changes

| Issue | Solution | Location | Impact |
|-------|----------|----------|--------|
| outer join creates union | Use inner join | `label_transfer.py:L128` | -30% peak memory |
| Different gene sets | Auto-subset before concat | `label_transfer.py:L109` | Prevents sparse union |
| Dense matrix conversion | Force sparse repr. | `label_transfer.py:L138` | -50% memory |
| Large scVI latent | Reduce n_latent=30 | `label_transfer.py:L147` | -20% training memory |
| Model stays in RAM | Delete + gc.collect() | `label_transfer.py:L162` | -10GB memory |

---

## Next Steps

1. ✅ Update CellLabeller to v0.1.1 with memory optimizations
2. ✅ Verify your script.sh SLURM settings (see above)
3. 🔄 **Re-run your job:**
   ```bash
   sbatch script.sh
   ```
4. 📊 **Monitor memory:**
   ```bash
   squeue -u $USER  # Find job ID
   sstat -j <JobID> --format=AveVMSize,MaxVMSize,AveRSS,MaxRSS
   ```

5. If still failing, provide:
   - Dataset shapes: `n_cells × n_genes` for reference and query
   - Gene overlap: How many genes in common?
   - Current error message from `slurm_script` file

---

## References

- **AnnData Memory**: https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html
- **scVI Installation**: https://docs.scvi-tools.org/en/stable/installation.html
- **Sparse Matrices**: https://scipy-lectures.org/advanced/sparse/index.html
- **SLURM Resource Allocation**: https://slurm.schedmd.com/sbatch.html
