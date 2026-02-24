# Getting Started with CellLabeller Development

This guide helps you contribute to CellLabeller development and set up a development environment.

## Development Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CellLabeller.git
cd CellLabeller
```

### 2. Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n celllabeller python=3.9
conda activate celllabeller
```

### 3. Install in Development Mode

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or manually install dev tools
pip install -r requirements.txt
pip install pytest black flake8 sphinx
```

### 4. Verify Installation

```python
python -c "import celllabeller; print(celllabeller.__version__)"
```

## Project Structure

```
CellLabeller/
├── celllabeller/                    # Main package
│   ├── __init__.py
│   ├── label_transfer.py           # Core label transfer class
│   ├── feature_engineering.py      # Feature selection
│   ├── hyperparameter_tuning.py    # Model optimization
│   └── utils.py                    # Utility functions
├── tests/                          # Test files
│   ├── test_label_transfer.py
│   ├── test_feature_engineering.py
│   └── test_hyperparameter_tuning.py
├── tutorial_label_transfer.ipynb   # Tutorial notebook
├── setup.py                        # Package setup
├── requirements.txt                # Dependencies
├── README.md                       # Main documentation
├── API_DOCUMENTATION.md            # API reference
└── LICENSE                         # MIT License
```

## Code Style and Standards

### Python Style Guide

We follow PEP 8 guidelines. Use these tools for code formatting:

```bash
# Format code with Black
black celllabeller/

# Check code style with Flake8
flake8 celllabeller/ --max-line-length=100

# Check imports
python -m isort celllabeller/
```

### Docstring Format

All functions and classes must have docstrings following NumPy style:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.
    
    Longer description with more details if needed.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2
    
    Returns
    -------
    bool
        Description of return value
    
    Examples
    --------
    >>> result = my_function(10, "test")
    >>> print(result)
    True
    
    Notes
    -----
    Any additional notes about the function.
    """
    # Implementation
    return True
```

### Type Hints

Use type hints for better code clarity:

```python
from typing import Optional, Tuple, List
import numpy as np
import anndata as ad

def process_data(
    adata: ad.AnnData,
    n_features: int = 500,
    batch_key: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Process AnnData object."""
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_label_transfer.py

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=celllabeller tests/
```

### Writing Tests

```python
# tests/test_my_module.py
import pytest
import numpy as np
import anndata as ad
from celllabeller import MyClass

@pytest.fixture
def sample_adata():
    """Fixture providing sample AnnData object."""
    X = np.random.rand(100, 50)
    adata = ad.AnnData(X)
    adata.obs["cell_type"] = np.random.choice(["A", "B", "C"], 100)
    return adata

def test_my_function(sample_adata):
    """Test my_function with sample data."""
    result = MyClass(sample_adata).do_something()
    assert result is not None
    assert len(result) == 100

def test_invalid_input():
    """Test handling of invalid input."""
    with pytest.raises(ValueError):
        MyClass(None)
```

## Common Development Tasks

### Adding a New Feature

1. **Create a feature branch:**
```bash
git checkout -b feature/my-new-feature
```

2. **Implement the feature** with proper docstrings and type hints

3. **Write tests** for the new feature

4. **Run tests and linting:**
```bash
pytest tests/
black celllabeller/
flake8 celllabeller/
```

5. **Update documentation** if needed

6. **Submit a pull request**

### Updating Dependencies

1. **Update requirements.txt:**
```bash
pip freeze > requirements.txt
```

2. **Update setup.py** if adding new required packages

3. **Test with updated dependencies:**
```bash
pip install -r requirements.txt
pytest tests/
```

### Building Documentation

If using Sphinx:

```bash
cd docs/
make html
# Open _build/html/index.html in browser
```

## Performance Optimization

### Profiling Code

```python
import cProfile
import pstats

# Profile a function
profiler = cProfile.Profile()
profiler.enable()

# Your code here
my_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### Memory Profiling

```bash
pip install memory-profiler
python -m memory_profiler script.py
```

## Git Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/description
```

### 2. Make Changes and Commit

```bash
git add celllabeller/
git commit -m "Add feature: description"
```

### 3. Keep Branch Updated

```bash
git fetch origin
git rebase origin/main
```

### 4. Push and Create Pull Request

```bash
git push origin feature/description
```

Then create a PR on GitHub.

## Bug Reporting

When reporting bugs, include:

1. **Minimal reproducible example:**
```python
import anndata as ad
import numpy as np
from celllabeller import CellTypeLabelTransfer

# Minimal code that reproduces the bug
X = np.random.rand(10, 5)
adata = ad.AnnData(X)
# ... rest of code
```

2. **Expected behavior**
3. **Actual behavior**
4. **Environment information:**
```bash
python --version
pip list | grep -E "celllabeller|scanpy|scvi|xgboost|torch"
```

## Documentation

### Updating README

- Keep the quick start section concise
- Add examples for common use cases
- Update references section for new citations

### Adding Examples

Create example scripts in `examples/` directory:

```python
# examples/basic_workflow.py
"""Basic label transfer workflow."""

import anndata as ad
from celllabeller import CellTypeLabelTransfer, FeatureEngineer, XGBoostTuner

# Load data
adata_ref = ad.read_h5ad("reference.h5ad")
adata_query = ad.read_h5ad("query.h5ad")

# ... rest of example
```

## Release Process

### Version Numbering

Use semantic versioning: `MAJOR.MINOR.PATCH`

- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Creating a Release

1. **Update version** in `celllabeller/__init__.py`
2. **Update CHANGELOG.md** with changes
3. **Commit changes:**
```bash
git add -A
git commit -m "Version bump to X.Y.Z"
git tag -a vX.Y.Z -m "Release version X.Y.Z"
```
4. **Push to GitHub:**
```bash
git push origin main --tags
```

## Support and Questions

- Open an GitHub issue for bugs and feature requests
- Check existing issues before creating new ones
- Use clear, descriptive titles for issues
- Include relevant code snippets and error messages

## Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows PEP 8 style guide
- [ ] All functions have docstrings
- [ ] Type hints are used
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No unused imports
- [ ] Commit messages are clear
- [ ] No hardcoded paths or credentials

## Additional Resources

- [Python Documentation](https://docs.python.org/3/)
- [NumPy Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Git Documentation](https://git-scm.com/doc)

## Questions?

Feel free to:
- Open an issue on GitHub
- Check existing documentation
- Review existing code for patterns

Happy coding! 🎉
