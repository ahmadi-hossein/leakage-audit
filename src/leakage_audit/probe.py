"""Acquisition-bias probe.

If a dataset was collected differently per class (different device,
compression, framing or site), crude image statistics can carry class
signal even though they say nothing about the content. This probe trains
a logistic regression on 11 such statistics only. Accuracy well above
chance means the dataset itself leaks class information through its
acquisition pipeline, and even a leakage-free split can overstate how
well a model would do on data from another source.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import Dataset

FEATURE_NAMES = [
    "mean_r",
    "mean_g",
    "mean_b",
    "overall_std",
    "border_mean",
    "border_std",
    "centre_mean",
    "centre_std",
    "file_size_kb",
    "width",
    "height",
]


def image_statistics(path) -> list[float]:
    """Compute the 11 crude statistics for one image.

    Border means the outer 10 percent frame of the image; centre means
    the middle 50 percent by area. Means and standard deviations are on
    grayscale values scaled to [0, 1].
    """
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
        file_size_kb = path.stat().st_size / 1024.0
    height, width = rgb.shape[0], rgb.shape[1]
    means = rgb.mean(axis=(0, 1)) / 255.0
    gray = rgb.mean(axis=2) / 255.0
    overall_std = float(gray.std())

    bh, bw = max(1, height // 10), max(1, width // 10)
    border = np.concatenate(
        [
            gray[:bh, :].ravel(),
            gray[-bh:, :].ravel(),
            gray[:, :bw].ravel(),
            gray[:, -bw:].ravel(),
        ]
    )
    ch, cw = height // 4, width // 4
    centre = gray[ch : height - ch, cw : width - cw]
    return [
        float(means[0]),
        float(means[1]),
        float(means[2]),
        overall_std,
        float(border.mean()),
        float(border.std()),
        float(centre.mean()),
        float(centre.std()),
        float(file_size_kb),
        float(width),
        float(height),
    ]


def run_probe(dataset: Dataset, seed: int = 42, cv_folds: int = 5) -> dict:
    """Run the acquisition-bias probe and return a summary dict.

    Uses stratified cross-validation with out-of-fold predictions when
    the smallest class has at least cv_folds members; otherwise falls
    back to a single stratified 70/30 holdout.
    """
    X = np.array([image_statistics(p) for p in dataset.files])
    y = dataset.labels
    classes, counts = np.unique(y, return_counts=True)
    majority_rate = float(counts.max() / counts.sum())
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    )
    if counts.min() >= cv_folds:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        pred = cross_val_predict(model, X, y, cv=cv)
        method = f"stratified {cv_folds}-fold cross-validation"
    else:
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(y))
        train_pos, test_pos = train_test_split(
            idx, test_size=0.30, random_state=seed, stratify=y
        )
        model.fit(X[train_pos], y[train_pos])
        pred = model.predict(X[test_pos])
        accuracy = float((pred == y[test_pos]).mean())
        return {
            "method": "stratified 70/30 holdout",
            "n_images": int(len(y)),
            "n_features": len(FEATURE_NAMES),
            "features": FEATURE_NAMES,
            "probe_accuracy": round(accuracy, 4),
            "majority_class_rate": round(majority_rate, 4),
            "chance_level": round(1.0 / len(classes), 4),
        }
    accuracy = float((pred == y).mean())
    return {
        "method": method,
        "n_images": int(len(y)),
        "n_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "probe_accuracy": round(accuracy, 4),
        "majority_class_rate": round(majority_rate, 4),
        "chance_level": round(1.0 / len(classes), 4),
    }
