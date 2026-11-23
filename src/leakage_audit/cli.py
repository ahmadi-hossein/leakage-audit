"""Command line interface for leakage-audit.

Subcommands:
  audit       Run the duplicate audit, check a random split, write reports.
  check-split Measure the leakage of a random stratified split.
  make-split  Write a leakage-free cluster-aware split as CSV files.
  probe       Run the acquisition-bias probe.

Every subcommand writes a JSON summary and prints a readable report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import DEFAULT_PHASH_THRESHOLD, audit_dataset
from .probe import run_probe
from .report import audit_markdown, pct
from .split import (
    check_random_split,
    cluster_aware_split,
    measure_split_leakage,
    write_split_csvs,
)


def _write_json(obj: dict, path: str | None) -> None:
    if path:
        Path(path).write_text(json.dumps(obj, indent=2))
        print(f"JSON summary written to {path}")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("data", help="dataset root with one subfolder per class")
    p.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_PHASH_THRESHOLD,
        help="pHash Hamming distance threshold for near duplicates "
        f"(default {DEFAULT_PHASH_THRESHOLD})",
    )
    p.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p.add_argument("--out", default=None, help="path for the JSON summary")


def cmd_audit(args: argparse.Namespace) -> int:
    dataset, cluster_of, summary = audit_dataset(args.data, args.threshold)
    split_check = check_random_split(dataset, cluster_of, seed=args.seed)
    combined = {"audit": summary, "random_split_check": split_check}
    report_path = args.report or "audit_report.md"
    Path(report_path).write_text(
        audit_markdown(summary, split_check=split_check)
    )
    print(f"Dataset: {summary['dataset']}")
    print(f"Images: {summary['n_images']}  Classes: {summary['class_counts']}")
    print(
        f"Exact-duplicate groups: {summary['exact_dup_groups']} "
        f"({summary['exact_dup_images']} images)"
    )
    print(
        f"Duplicate clusters: {summary['dup_clusters']} covering "
        f"{summary['images_in_dup_clusters']} images "
        f"({pct(summary['images_in_dup_clusters_rate'])})"
    )
    print(
        f"Random split: {split_check['test_cluster_leakage']} of "
        f"{split_check['test_size']} test images leak into train "
        f"({pct(split_check['test_cluster_leakage_rate'])}), "
        f"{split_check['test_exact_leakage']} of them exact copies"
    )
    print(f"Markdown report written to {report_path}")
    _write_json(combined, args.out or "audit_summary.json")
    return 0


def cmd_check_split(args: argparse.Namespace) -> int:
    dataset, cluster_of, _ = audit_dataset(args.data, args.threshold)
    result = check_random_split(dataset, cluster_of, seed=args.seed)
    print(
        f"Random stratified split (seed {args.seed}): "
        f"train {result['train_size']}, val {result['val_size']}, "
        f"test {result['test_size']}"
    )
    print(
        f"Test images with a duplicate-cluster mate in train: "
        f"{result['test_cluster_leakage']} "
        f"({pct(result['test_cluster_leakage_rate'])})"
    )
    print(f"Test images that are exact copies of a train image: "
          f"{result['test_exact_leakage']}")
    _write_json(result, args.out or "split_check.json")
    return 0


def cmd_make_split(args: argparse.Namespace) -> int:
    dataset, cluster_of, _ = audit_dataset(args.data, args.threshold)
    train, val, test = cluster_aware_split(
        dataset, cluster_of, seed=args.seed
    )
    out_dir = args.out_dir or "splits"
    written = write_split_csvs(
        dataset,
        cluster_of,
        {"train": train, "val": val, "test": test},
        out_dir,
    )
    check = measure_split_leakage(dataset, cluster_of, train, test)
    print(f"Cluster-aware split written to {out_dir}:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    print(
        f"Sizes: train {len(train)}, val {len(val)}, test {len(test)}; "
        f"test leakage into train: {check['test_cluster_leakage']} "
        f"(must be 0)"
    )
    _write_json(
        {
            "sizes": {
                "train": int(len(train)),
                "val": int(len(val)),
                "test": int(len(test)),
            },
            "leakage_check": check,
            "files": written,
        },
        args.out or "make_split.json",
    )
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    from .core import load_dataset

    dataset = load_dataset(args.data)
    result = run_probe(dataset, seed=args.seed)
    print(f"Acquisition-bias probe ({result['method']})")
    print(
        f"Logistic regression on {result['n_features']} image statistics: "
        f"accuracy {pct(result['probe_accuracy'])}"
    )
    print(
        f"Majority-class baseline: {pct(result['majority_class_rate'])}; "
        f"chance: {pct(result['chance_level'])}"
    )
    _write_json(result, args.out or "probe.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="leakage-audit",
        description="Two-stage audit for image classification datasets: "
        "duplicate-leakage detection and an acquisition-bias probe.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="run the full duplicate audit")
    _add_common(p_audit)
    p_audit.add_argument(
        "--report", default=None, help="path for the Markdown report"
    )
    p_audit.set_defaults(func=cmd_audit)

    p_check = sub.add_parser(
        "check-split", help="measure leakage of a random stratified split"
    )
    _add_common(p_check)
    p_check.set_defaults(func=cmd_check_split)

    p_make = sub.add_parser(
        "make-split", help="write a leakage-free cluster-aware split"
    )
    _add_common(p_make)
    p_make.add_argument(
        "--out-dir", default=None, help="directory for train/val/test CSVs"
    )
    p_make.set_defaults(func=cmd_make_split)

    p_probe = sub.add_parser("probe", help="run the acquisition-bias probe")
    p_probe.add_argument("data", help="dataset root with one subfolder per class")
    p_probe.add_argument("--seed", type=int, default=42)
    p_probe.add_argument("--out", default=None, help="path for the JSON summary")
    p_probe.set_defaults(func=cmd_probe)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
