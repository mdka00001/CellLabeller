from setuptools import setup, find_packages

setup(
    name="celllabeller",
    version="0.1.0",
    description="XGBoost-based end-to-end single-cell annotation tool",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "anndata>=0.9",
        "scanpy>=1.9",
        "scvi-tools>=1.0",
        "xgboost>=1.7",
        "scikit-learn>=1.2",
        "numpy>=1.23",
        "pandas>=1.5",
        "scipy>=1.9",
        "joblib>=1.2",
        "matplotlib>=3.6",
        "seaborn>=0.12",
    ],
    extras_require={
        "optuna": ["optuna>=3.0"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
)
