"""Split leakage measurement and leakage-free cluster-aware splitting."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .core import Dataset


def random_stratified_split(
    labels: np.ndarray,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce a standard random stratified split (default 70/15/15)."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    idx = np.arange(len(labels))
    rest, test = train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=labels
    )
    val_share_of_rest = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        rest,
        test_size=val_share_of_rest,
        random_state=seed,
        stratify=labels[rest],
    )
    return train, val, test


def measure_split_leakage(
    dataset: Dataset,
    cluster_of: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    """Measure how much of the test set leaks into the training set.

    A test image leaks if another image from its duplicate cluster is in
    the training set. Exact leakage is the stricter subset where the MD5
    checksum itself appears in training.
    """
    train_clusters = set(cluster_of[train_idx].tolist())
    leaked = [i for i in test_idx if cluster_of[i] in train_clusters]
    train_md5 = {dataset.md5s[i] for i in train_idx}
    exact = [i for i in test_idx if dataset.md5s[i] in train_md5]
    n_test = len(test_idx)
    return {
        "train_size": int(len(train_idx)),
        "test_size": int(n_test),
        "test_cluster_leakage": int(len(leaked)),
        "test_cluster_leakage_rate": round(len(leaked) / n_test, 4) if n_test else 0.0,
        "test_exact_leakage": int(len(exact)),
    }


def check_random_split(
    dataset: Dataset,
    cluster_of: np.ndarray,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict:
    """Measure the leakage of a random stratified split of the dataset."""
    train, val, test = random_stratified_split(
        dataset.labels,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    result = measure_split_leakage(dataset, cluster_of, train, test)
    result.update(
        {
            "seed": seed,
            "split": "random stratified",
            "val_size": int(len(val)),
            "fractions": {
                "train": train_frac,
                "val": val_frac,
                "test": test_frac,
            },
        }
    )
    return result


def cluster_aware_split(
    dataset: Dataset,
    cluster_of: np.ndarray,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split whole duplicate clusters into train/val/test (leakage-free).

    Uses group shuffling on the cluster ids. Because clusters can be
    large, exact fractions and exact class balance are approximate;
    leakage across the boundary is exactly zero by construction.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    idx = np.arange(len(dataset))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    rest, test = next(gss.split(idx, groups=cluster_of))
    val_share_of_rest = val_frac / (train_frac + val_frac)
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=val_share_of_rest, random_state=seed
    )
    train_sub, val_sub = next(
        gss2.split(rest, groups=cluster_of[rest])
    )
    return rest[train_sub], rest[val_sub], test


def write_split_csvs(
    dataset: Dataset,
    cluster_of: np.ndarray,
    splits: dict[str, np.ndarray],
    out_dir: str | Path,
) -> dict[str, str]:
    """Write train/val/test CSVs with path, label and cluster columns."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, indices in splits.items():
        path = out_dir / f"{name}.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "label", "cluster_id"])
            for i in sorted(indices.tolist()):
                writer.writerow(
                    [
                        str(dataset.files[i].relative_to(dataset.root)),
                        dataset.class_names[dataset.labels[i]],
                        int(cluster_of[i]),
                    ]
                )
        written[name] = str(path)
    return written
