"""Core audit logic: hashing, duplicate clustering, and audit summaries.

The approach follows a two-stage audit of a public medical imaging dataset:

1. Exact duplicates are found with MD5 checksums of the raw file bytes.
2. Near duplicates are found with perceptual hashes (pHash, hash_size=8).
   Pairs whose Hamming distance is at most a threshold (default 5) are
   merged, together with the exact-duplicate groups, into clusters using
   union-find. A cluster is the unit that must never be split across
   train and test.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_PHASH_THRESHOLD = 5


@dataclass
class Dataset:
    """An image classification dataset laid out as one subfolder per class."""

    root: Path
    files: list[Path]
    labels: np.ndarray  # integer class index per file
    class_names: list[str]
    md5s: list[str] = field(default_factory=list)
    phashes: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.files)

    def compute_hashes(self) -> None:
        """Compute MD5 checksums and perceptual hashes for every image."""
        n = len(self.files)
        self.md5s = []
        phashes = np.zeros(n, dtype=np.uint64)
        for i, path in enumerate(self.files):
            self.md5s.append(hashlib.md5(path.read_bytes()).hexdigest())
            with Image.open(path) as img:
                h = imagehash.phash(img.convert("RGB"), hash_size=8)
            phashes[i] = int(str(h), 16)
        self.phashes = phashes


def load_dataset(root: str | Path) -> Dataset:
    """Load a dataset from a folder with one subfolder per class.

    Subfolders are taken in sorted order; the class name is the folder
    name. Raises ValueError if fewer than two class folders contain
    images.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    files: list[Path] = []
    labels: list[int] = []
    class_names: list[str] = []
    for class_dir in class_dirs:
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            continue
        idx = len(class_names)
        class_names.append(class_dir.name)
        files.extend(images)
        labels.extend([idx] * len(images))
    if len(class_names) < 2:
        raise ValueError(
            f"expected at least two class subfolders with images under {root}"
        )
    return Dataset(
        root=root,
        files=files,
        labels=np.array(labels),
        class_names=class_names,
    )


class UnionFind:
    """Union-find with path halving, as used for duplicate clustering."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def exact_duplicate_groups(md5s: list[str]) -> list[list[int]]:
    """Return groups of indices that share an MD5 checksum (size > 1)."""
    by_hash: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(md5s):
        by_hash[m].append(i)
    return [g for g in by_hash.values() if len(g) > 1]


def hamming_pairs_and_union(
    phashes: np.ndarray, uf: UnionFind, threshold: int
) -> int:
    """Union all pairs with Hamming distance <= threshold; return pair count.

    Distances are computed in row chunks of 1024 to bound memory, counting
    bits by repeated shift-and-add over the XOR of the hash values.
    """
    n = len(phashes)
    pairs = 0
    for s in range(0, n, 1024):
        chunk = phashes[s : s + 1024]
        xor = np.bitwise_xor(chunk[:, None], phashes[None, :])
        dist = np.zeros(xor.shape, dtype=np.uint8)
        x = xor.copy()
        while x.any():
            dist += (x & 1).astype(np.uint8)
            x >>= 1
        ii, jj = np.where(dist <= threshold)
        for a, b in zip(ii, jj):
            gi, gj = s + int(a), int(b)
            if gi < gj:
                pairs += 1
                uf.union(gi, gj)
    return pairs


def build_clusters(
    dataset: Dataset, threshold: int = DEFAULT_PHASH_THRESHOLD
) -> tuple[np.ndarray, dict]:
    """Cluster exact and near duplicates. Returns (cluster_id per image, stats).

    Requires dataset.compute_hashes() to have been called.
    """
    if dataset.phashes is None or not dataset.md5s:
        raise ValueError("call dataset.compute_hashes() before build_clusters()")
    n = len(dataset)
    uf = UnionFind(n)
    exact_groups = exact_duplicate_groups(dataset.md5s)
    for group in exact_groups:
        for j in group[1:]:
            uf.union(group[0], j)
    near_pairs = hamming_pairs_and_union(dataset.phashes, uf, threshold)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    # Stable, compact cluster ids ordered by smallest member index.
    ordered = sorted(clusters.values(), key=lambda g: g[0])
    cluster_of = np.empty(n, dtype=np.int64)
    for cid, group in enumerate(ordered):
        for i in group:
            cluster_of[i] = cid
    multi = [g for g in ordered if len(g) > 1]
    cross_class = sum(
        1 for g in multi if len(set(dataset.labels[g].tolist())) > 1
    )
    stats = {
        "exact_dup_groups": len(exact_groups),
        "exact_dup_images": int(sum(len(g) for g in exact_groups)),
        "near_dup_pairs": int(near_pairs),
        "phash_threshold": int(threshold),
        "dup_clusters": len(multi),
        "images_in_dup_clusters": int(sum(len(g) for g in multi)),
        "cross_class_clusters": int(cross_class),
        "n_clusters_total": len(ordered),
    }
    return cluster_of, stats


def audit_dataset(
    root: str | Path, threshold: int = DEFAULT_PHASH_THRESHOLD
) -> tuple[Dataset, np.ndarray, dict]:
    """Run the full duplicate audit and return (dataset, cluster_of, summary)."""
    dataset = load_dataset(root)
    dataset.compute_hashes()
    cluster_of, stats = build_clusters(dataset, threshold=threshold)
    n = len(dataset)
    class_counts = {
        name: int((dataset.labels == i).sum())
        for i, name in enumerate(dataset.class_names)
    }
    summary = {
        "dataset": str(dataset.root),
        "n_images": int(n),
        "class_counts": class_counts,
        **stats,
        "images_in_dup_clusters_rate": round(
            stats["images_in_dup_clusters"] / n, 4
        )
        if n
        else 0.0,
    }
    return dataset, cluster_of, summary
