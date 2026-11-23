"""Tests for leakage-audit using small synthetic images."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from leakage_audit.core import audit_dataset, build_clusters, load_dataset
from leakage_audit.probe import run_probe
from leakage_audit.split import (
    cluster_aware_split,
    measure_split_leakage,
)


def _image(path: Path, seed: int, size: int = 96) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=3)
    arr = np.zeros((size, size, 3), dtype=np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    for c in range(3):
        arr[:, :, c] = base[c] + 25 * (xx / size)
    cx, cy = rng.integers(20, size - 20, size=2)
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 15**2
    arr[mask] += 50
    arr += rng.normal(0, 3, arr.shape)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path, quality=92)


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for cls in ("alpha", "beta"):
        folder = root / cls
        folder.mkdir(parents=True)
        for i in range(10):
            _image(folder / f"{cls}_{i}.jpg", seed=hash((cls, i)) % 10000)
    # Plant two exact duplicates in alpha and one in beta.
    shutil.copyfile(root / "alpha" / "alpha_0.jpg", root / "alpha" / "alpha_0_dup.jpg")
    shutil.copyfile(root / "alpha" / "alpha_1.jpg", root / "alpha" / "alpha_1_dup.jpg")
    shutil.copyfile(root / "beta" / "beta_0.jpg", root / "beta" / "beta_0_dup.jpg")
    # Plant a near duplicate: a slightly resized copy.
    src = Image.open(root / "beta" / "beta_1.jpg")
    src.resize((84, 84)).resize((96, 96)).save(root / "beta" / "beta_1_near.jpg", quality=92)
    return root


def test_load_dataset(dataset_dir):
    ds = load_dataset(dataset_dir)
    assert len(ds) == 24
    assert ds.class_names == ["alpha", "beta"]


def test_exact_duplicate_detection(dataset_dir):
    _, _, summary = audit_dataset(dataset_dir)
    assert summary["exact_dup_groups"] == 3
    assert summary["exact_dup_images"] == 6


def test_cluster_formation(dataset_dir):
    ds = load_dataset(dataset_dir)
    ds.compute_hashes()
    cluster_of, stats = build_clusters(ds)
    assert stats["dup_clusters"] >= 3
    # The resized near duplicate joins its source's cluster.
    names = [p.name for p in ds.files]
    src = names.index("beta_1.jpg")
    near = names.index("beta_1_near.jpg")
    assert cluster_of[src] == cluster_of[near]
    # Exact duplicates share a cluster.
    a0 = names.index("alpha_0.jpg")
    a0d = names.index("alpha_0_dup.jpg")
    assert cluster_of[a0] == cluster_of[a0d]


def test_cluster_aware_split_has_zero_leakage(dataset_dir):
    ds, cluster_of, _ = audit_dataset(dataset_dir)
    train, val, test = cluster_aware_split(ds, cluster_of, seed=0)
    assert len(train) + len(val) + len(test) == len(ds)
    check = measure_split_leakage(ds, cluster_of, train, test)
    assert check["test_cluster_leakage"] == 0
    assert check["test_exact_leakage"] == 0


def test_probe_runs(dataset_dir):
    ds = load_dataset(dataset_dir)
    result = run_probe(ds)
    assert 0.0 <= result["probe_accuracy"] <= 1.0
    assert result["n_features"] == 11
