"""leakage-audit: a two-stage audit for image classification datasets.

Stage 1 detects exact and near duplicates (MD5 plus perceptual hashing),
clusters them, and measures how much of a random split leaks across the
train/test boundary. Stage 2 runs an acquisition-bias probe: a logistic
regression on crude image statistics that should not carry class signal.
"""

from .core import Dataset, audit_dataset, load_dataset

__version__ = "0.1.0"
__all__ = ["Dataset", "audit_dataset", "load_dataset", "__version__"]
