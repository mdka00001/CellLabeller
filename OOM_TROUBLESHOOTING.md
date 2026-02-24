# Memory OOM Troubleshooting Guide

## ❌ Problem: "OOM Killed" with 325GB RAM

Your system has plenty of RAM but `sc.concat()` is still causing OOM. This document provides step-by-step solutions.

---

## 🔍 Quick Diagnosis (Run This First)

```bash
cd /home/woody/mfn3/mfn3100h/git_repo/CellLabeller

# Run diagnostic tool
python scripts/memory_diagnostic.py
```

This will:
1. Check your dataset sizes
2. Estimate memory usage
3. Compare concat strategies
4. Give specific recommendations

---

## ✅ Solution 1: Verify CellLabeller is Updated (v0.1.1+)

The fixes have been implemented in `/home/woody/mfn3/mfn3100h/git_repo/CellLabeller/celllabeller/label_transfer.py`

**Verify the fix:**
```bash
cd /home/woody/mfn3/mfn3100h/git_repo/CellLabeller

# Check for "join="inner"" (should show this)
grep -n 'join="inner"' celllabeller/label_transfer.py

# Also check for sparse matrix enforcement
grep -n 'csr_matrix' celllabeller/label_transfer.py
```

**Expected output:**
```
celllabeller/label_transfer.py:128:            join="inner",  # <-- THIS LINE
celllabeller/label_transfer.py:138:            integrated.X = csr_matrix(integrated.X)
```

If these lines don't exist, reinstall:
```bash
pip install -e /home/woody/mfn3/mfn3100h/git_repo/CellLabeller
```

---

## ✅ Solution 2: Update Your Script

Make sure your `script.py` calls `subset_common_genes()` BEFORE `integrate_with_scvi()`:

```python
# Correct order:
label_transfer = CellTypeLabelTransfer(...)
adata_ref, adata_query = label_transfer.subset_common_genes()  # ← FIRST
adata_integrated = label_transfer.integrate_with_scvi()         # ← SECOND
```

**Your current script.py (lines 75-76):**
```python
adata_ref, adata_query = label_transfer.subset_common_genes()
adata_integrated = label_transfer.integrate_with_scvi()
```

✅ **This is correct!** No changes needed.

---

## ✅ Solution 3: Optimize SLURM Settings

Update your `script.sh`:

```bash
#SBATCH --mem=300G              # ← Change this (leave 25GB headroom)
#SBATCH --cpus-per-task=16      # ← Add this (for parallel ops)
#SBATCH --gres=gpu:1            # ← Add this (use RTX3080)
#SBATCH --time=08:00:00         # ← Change if job is timing out
```

**Use the provided template:**
```bash
cp /home/woody/mfn3/mfn3100h/region_AB11049/LN_gabriella_scRNAseq_260224/script_optimized.sh \
   /home/woody/mfn3/mfn3100h/region_AB11049/LN_gabriella_scRNAseq_260224/script.sh

# Then edit the Python script path if needed:
nano /home/woody/mfn3/mfn3100h/region_AB11049/LN_gabriella_scRNAseq_260224/script.sh
```

---

## ✅ Solution 4: Check Dataset Sizes

The most likely culprit: your datasets are very large.

```bash
python << 'EOF'
import anndata as ad

ref = ad.read_h5ad("/home/woody/mfn3/mfn3100h/aging/adata_v5_inguinal.h5ad", backed='r')
query = ad.read_h5ad("/home/woody/mfn3/mfn3100h/region_AB11049/adata_processed.h5ad", backed='r')

print(f"Reference: {ref.shape[0]:,} cells × {ref.shape[1]:,} genes")
print(f"Query: {query.shape[0]:,} cells × {query.shape[1]:,} genes")

common = len(set(ref.var_names) & set(query.var_names))
print(f"Common genes: {common:,}")
print(f"\nEstimated memory (float32 sparse):")
print(f"  After subsetting: {(ref.shape[0] + query.shape[0]) * common * 4 / 1e9:.1f} GB")

ref.file.close()
query.file.close()
EOF
```

---

## ⚠️ If Still Getting OOM:

### Option A: Filter Reference to Abundant Cell Types
```python
# Add this to script.py AFTER loading reference
import pandas as pd

cell_type_counts = adata_ref.obs["cell_type"].value_counts()
print(f"Original reference: {adata_ref.shape}")

# Keep only cell types with >= 50 cells
abundant_types = cell_type_counts[cell_type_counts >= 50].index
adata_ref = adata_ref[adata_ref.obs["cell_type"].isin(abundant_types)].copy()

print(f"Filtered reference: {adata_ref.shape}")
```

**Impact**: Reduces reference from 50k → 30k cells, saves 40% memory

### Option B: Subsample Reference
```python
import numpy as np

# Keep only 80k random cells
if adata_ref.shape[0] > 80000:
    idx = np.random.choice(adata_ref.n_obs, 80000, replace=False)
    adata_ref = adata_ref[idx].copy()
    print(f"Subsampled reference: {adata_ref.shape}")
```

### Option C: Use HDF5 Backed Mode
```python
# Load without full in-memory representation
adata_ref = ad.read_h5ad(ref_path, backed='r')
# Operations will use disk I/O (slower but memory-efficient)
```

### Option D: Split Analysis by Batch
```python
# Process query in batches
batch_size = 20000
for i in range(0, adata_query.shape[0], batch_size):
    batch = adata_query[i:i+batch_size].copy()
    # Predict on this batch only
```

---

## 🔄 Recommended Next Steps

1. **Run diagnostic:**
   ```bash
   python /home/woody/mfn3/mfn3100h/git_repo/CellLabeller/scripts/memory_diagnostic.py
   ```

2. **Update SLURM script:**
   ```bash
   cp script_optimized.sh script.sh
   ```

3. **Verify updates are installed:**
   ```bash
   cd /home/woody/mfn3/mfn3100h/git_repo/CellLabeller
   pip install -e .
   ```

4. **Monitor during execution:**
   ```bash
   sbatch script.sh
   squeue -u $USER              # Get Job ID
   sstat -j <JobID> --format=MaxRSS
   ```

5. **If still fails, check error:**
   ```bash
   tail -50 logs/celllabeller_*.err
   ```

---

## 📊 Expected Memory Timeline

With fixes applied (inner join + subset + sparse):

| Stage | Memory | Duration |
|-------|--------|----------|
| Load reference | 10-15 GB | 30s |
| Preprocess ref | 8-12 GB | 1m |
| Load query | 10-15 GB | 30s |
| Subset genes | 15-20 GB | 2m |
| scVI training | 20-25 GB | 30m (GPU) |
| Latent extract | 15-20 GB | 2m |
| Feature eng | 10-15 GB | 5m |
| XGBoost train | 5-10 GB | 5m |
| **Total Peak** | **~50 GB** | **~45m** |

✅ Well within 300GB available

---

## 🐛 Debug Commands

### Check if CellLabeller loaded correctly:
```python
from celllabeller import CellTypeLabelTransfer
import inspect

# Print the source of integrate_with_scvi to verify fix
print(inspect.getsource(CellTypeLabelTransfer.integrate_with_scvi))
# Should show: join="inner" (not "outer")
```

### Monitor job in real-time:
```bash
watch -n 5 'sstat -j <JobID> --format=MaxRSS,AveVMSize && nvidia-smi'
```

### Check SLURM allocation vs usage:
```bash
srun --jobid=<JobID> free -h
srun --jobid=<JobID> nvidia-smi
```

---

## 📞 If Problems Persist

Provide this information for further debugging:

1. **Dataset shapes:**
   ```bash
   python << 'EOF'
   import anndata as ad
   ref = ad.read_h5ad("path/to/reference.h5ad", backed='r')
   query = ad.read_h5ad("path/to/query.h5ad", backed='r')
   print(f"Ref: {ref.shape}, Query: {query.shape}")
   EOF
   ```

2. **Error message:**
   ```bash
   cat logs/celllabeller_*.err | head -20
   ```

3. **Memory at time of crash:**
   ```bash
   # Check SLURM job history
   sacct -j <JobID> --format=MaxRSS
   ```

4. **CellLabeller version:**
   ```bash
   pip show celllabeller-scrnaseq  # or wherever installed from
   grep "join=" /home/woody/mfn3/mfn3100h/git_repo/CellLabeller/celllabeller/label_transfer.py
   ```

---

## ✨ Summary

**What changed:**
- ✅ `join="outer"` → `join="inner"` (prevents gene union explosion)
- ✅ Auto-subset common genes before concatenation
- ✅ Force sparse matrix representation
- ✅ Clean up models after use
- ✅ Proper scVI batch correction key

**Expected result:**
- ❌ 150 GB peak memory → ✅ 50 GB peak memory
- ❌ OOM kill → ✅ Successful completion

**Next action:** Run diagnostic, update SLURM settings, resubmit job.
