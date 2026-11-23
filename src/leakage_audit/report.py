"""Markdown report generation for audit results."""
from __future__ import annotations


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def audit_markdown(summary: dict, split_check: dict | None = None,
                   probe: dict | None = None) -> str:
    """Render a Markdown report from audit (and optional split/probe) results."""
    lines: list[str] = []
    lines.append("# Leakage audit report")
    lines.append("")
    lines.append(f"Dataset: `{summary['dataset']}`")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Images: {summary['n_images']}")
    for name, count in summary["class_counts"].items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## Stage 1: duplicate-leakage audit")
    lines.append("")
    lines.append(
        f"- Exact-duplicate groups (MD5): {summary['exact_dup_groups']} "
        f"covering {summary['exact_dup_images']} images"
    )
    lines.append(
        f"- Near-duplicate pairs (pHash Hamming distance "
        f"<= {summary['phash_threshold']}): {summary['near_dup_pairs']}"
    )
    lines.append(
        f"- Duplicate clusters (exact + near): {summary['dup_clusters']} "
        f"covering {summary['images_in_dup_clusters']} images "
        f"({pct(summary['images_in_dup_clusters_rate'])} of the dataset)"
    )
    lines.append(
        f"- Clusters mixing classes: {summary['cross_class_clusters']}"
    )
    lines.append("")
    if split_check is not None:
        lines.append("## Random split leakage check")
        lines.append("")
        lines.append(
            f"- Split: {split_check['split']}, seed {split_check['seed']}, "
            f"train {split_check['train_size']}, "
            f"val {split_check['val_size']}, test {split_check['test_size']}"
        )
        lines.append(
            f"- Test images with a duplicate-cluster mate in train: "
            f"{split_check['test_cluster_leakage']} of "
            f"{split_check['test_size']} "
            f"({pct(split_check['test_cluster_leakage_rate'])})"
        )
        lines.append(
            f"- Test images that are exact copies of a train image: "
            f"{split_check['test_exact_leakage']}"
        )
        lines.append("")
        lines.append(
            "A test set that shares duplicate clusters with the training "
            "set does not measure generalisation. Use a cluster-aware "
            "split (leakage-audit make-split) so that no cluster crosses "
            "the train/test boundary."
        )
        lines.append("")
    if probe is not None:
        lines.append("## Stage 2: acquisition-bias probe")
        lines.append("")
        lines.append(
            f"- Logistic regression on {probe['n_features']} crude image "
            f"statistics ({probe['method']}): accuracy "
            f"{pct(probe['probe_accuracy'])}"
        )
        lines.append(
            f"- Majority-class baseline: {pct(probe['majority_class_rate'])}; "
            f"chance level: {pct(probe['chance_level'])}"
        )
        lines.append("")
        lines.append(
            "Probe accuracy well above the majority baseline means the "
            "classes differ in how the images were acquired or processed, "
            "not only in their content. Treat headline accuracy with "
            "caution and validate on data from a different source."
        )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Clusters are the unit of splitting. If images carry patient or "
        "subject identifiers, group by those instead; duplicate clusters "
        "are a fallback when no such identifiers exist."
    )
    lines.append(
        "- The pHash threshold trades recall against precision: a lower "
        "threshold misses recompressed or cropped copies, a higher one "
        "merges genuinely different images. Inspect the largest clusters "
        "before trusting a threshold."
    )
    lines.append("")
    return "\n".join(lines)
