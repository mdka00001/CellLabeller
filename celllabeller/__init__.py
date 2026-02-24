"""CellLabeller: XGBoost-based end-to-end single-cell annotation tool.

Integrates a query and a reference AnnData using scVI-based shared latent
space, engineers features from gene expression and the latent space,
trains an XGBoost cell-type classifier on the reference, and transfers
labels to the query dataset.
"""

from .pipeline import CellLabeller  # noqa: F401

__version__ = "0.1.0"
__all__ = ["CellLabeller"]
