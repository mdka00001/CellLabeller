# Changelog

All notable changes to CellLabeller will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-02-24

### Added

#### Core Features
- **Cell Type Label Transfer Pipeline**: Complete end-to-end pipeline for transferring cell type labels from reference to query datasets
- **Gene Subsetting**: Automatic identification and subsetting of both datasets to common genes
- **scVI Integration**: Integration of reference and query datasets using scVI with configurable epochs (minimum 200 as required)
- **Flexible Feature Engineering**: User-configurable feature selection with three options:
  - Gene expression features only
  - scVI latent space features only
  - Combined genes and latent space features
- **XGBoost Hyperparameter Tuning**: Bayesian optimization using Optuna with 9 tunable hyperparameters
- **GPU/CPU Comparison**: Full support for both GPU and CPU acceleration with automatic comparison framework
- **Comprehensive Model Evaluation**: Accuracy, balanced accuracy, F1-score, and confusion matrix metrics

#### Modules
- `CellTypeLabelTransfer`: Main class for preprocessing and scVI integration
- `FeatureEngineer`: Flexible feature selection and engineering
- `XGBoostTuner`: Hyperparameter optimization and model training with GPU/CPU support
- `utils`: Utility functions for model loading and prediction

#### Documentation
- Comprehensive README with quick start guide
- API documentation with all module and class details
- Development guide for contributors
- Tutorial Jupyter notebook with complete working examples
- Setup.py and requirements.txt for easy installation
- MIT License

#### Examples and Utilities
- Ready-to-execute tutorial notebook (tutorial_label_transfer.ipynb)
- Utility functions for loading models and making predictions
- Automatic result saving and organization

### Features Implemented

✅ **Requirement 1**: Subset both datasets with common genes
- Automatic identification of genes present in both datasets
- Logging of genes removed from each dataset

✅ **Requirement 2**: Integrate with scVI (≥200 epochs)
- Full scVI integration pipeline
- Configurable epochs (minimum enforced to 200)
- Automatic latent space extraction
- Model saving for reproducibility

✅ **Requirement 3**: User-based feature engineering
- Three selectable feature types: genes, scvi_latent, combined
- Optional scVI component limiting
- Feature standardization

✅ **Requirement 4**: Hyperparameter testing on XGBoost with GPU/CPU options
- Bayesian optimization with Optuna
- Independent optimization for GPU and CPU
- 9 hyperparameters automatically tuned
- Graceful fallback if GPU unavailable

✅ **Requirement 5**: Score predictions on training and test data
- Accuracy, balanced accuracy, F1-score metrics
- Confusion matrices
- Per-class classification reports
- Prediction probabilities

✅ **Requirement 6**: Store results in local folder
- Organized results directory structure
- Models saved as pickles
- Hyperparameter results saved
- Trial histories as CSV
- Integrated data as HDF5

✅ **Requirement 7**: Tutorial Jupyter notebook
- Complete, ready-to-execute tutorial
- Clear instructions for each step
- Configurable parameters
- Visualization of results
- Next steps guidance

✅ **Requirement 8**: Documentation
- README with overview and quick start
- API documentation with all method details
- Development guide for contributors
- Docstrings in all modules
- Type hints throughout codebase

### Performance

- Efficient data processing with NumPy and Pandas
- GPU acceleration for both scVI and XGBoost
- Parallel hyperparameter optimization
- Memory-efficient feature selection

### Quality Assurance

- Type hints for all function signatures
- Comprehensive docstrings (NumPy style)
- Proper error handling and logging
- Consistent code style (PEP 8 compatible)

## Planned for Future Releases

### [0.2.0] - Planned
- [ ] Additional deep learning-based classifiers (neural networks)
- [ ] Cross-validation framework
- [ ] Batch effect correction improvements
- [ ] Integration with scArches for transfer learning
- [ ] Web API for model serving
- [ ] Docker containerization

### [0.3.0] - Planned
- [ ] Ensemble methods combining multiple models
- [ ] Active learning for uncertain cells
- [ ] Cell type annotation refinement
- [ ] Integration with CellTypist and other annotation tools

### Future Ideas
- [ ] Integration with other integration methods (Harmony, Seurat)
- [ ] Support for multi-modal data
- [ ] Interactive web interface for prediction
- [ ] Support for hierarchical cell type annotations
- [ ] Population-level transfer learning

## Known Issues

- GPU acceleration requires CUDA-capable GPU; graceful CPU fallback implemented
- Very large datasets (>1M cells) may require memory optimization
- scVI convergence may be slow on very sparse data

## Deprecation Notices

None at this time.

## Migration Guide

N/A - Initial release

## Contributors

- Primary developers: [Your Name]
- Contributors welcome! See [DEVELOPMENT.md](DEVELOPMENT.md)

## References

### Key Publications

1. **scVI**: Lopez, R., et al. (2018). "Deep generative modeling for single-cell transcriptomics." Nature Methods. [Link](https://doi.org/10.1038/s41592-018-0229-2)

2. **XGBoost**: Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. [Link](https://doi.org/10.1145/2939672.2939785)

3. **Scanpy**: Wolf, F. A., et al. (2018). "SCANPY, large-scale single-cell gene expression data analysis." Genome Biology. [Link](https://doi.org/10.1186/s13059-017-1382-2)

4. **AnnData**: Virshup, I., et al. (2021). "The scverse project provides Python infrastructure for spatial single-cell omics." Nature Biotechnology. [Link](https://doi.org/10.1038/s41587-023-01733-8)

5. **Optuna**: Akiba, T., et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. [Link](https://doi.org/10.1145/3292500.3330701)

---

For detailed information about each release, see the corresponding GitHub release page.
