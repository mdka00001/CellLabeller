from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="CellLabeller",
    version="0.1.0",
    author="Md. Adnan Karim",
    description="An XGBoost-based end-to-end single cell annotation tool using scVI integration and transfer learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CellLabeller",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "anndata>=0.7.0",
        "scanpy>=1.9.0",
        "scvi-tools>=0.17.0",
        "xgboost>=1.5.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
        "optuna>=2.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)
