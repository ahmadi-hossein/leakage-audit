"""Streamlit front end for the leakage-audit tool.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Or deploy free on Streamlit Community Cloud from the GitHub repo
(repo: ahmadi-hossein/leakage-audit, main file: app.py).

The app wraps the leakage_audit package in src/:
1. Duplicate audit - exact (MD5) and near-duplicate (pHash) clustering.
2. Split check - how much of a random split's test set leaks into train.
3. Leakage-free split - cluster-aware train/val/test CSVs you can download.
4. Acquisition-bias probe - logistic regression on 11 crude image
   statistics only; accuracy well above chance means the dataset itself
   carries class signal from how it was collected.
"""
from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "src"))

from leakage_audit.core import audit_dataset, load_dataset  # noqa: E402
from leakage_audit.probe import FEATURE_NAMES, run_probe  # noqa: E402
from leakage_audit.split import (  # noqa: E402
    check_random_split,
    cluster_aware_split,
    measure_split_leakage,
    write_split_csvs,
)

st.set_page_config(page_title="leakage-audit", page_icon="🔍", layout="wide")

st.title("leakage-audit")
st.markdown(
    "A two-stage audit for image classification datasets: "
    "**duplicate-leakage detection** and an **acquisition-bias probe**. "
    "Source: [github.com/ahmadi-hossein/leakage-audit]"
    "(https://github.com/ahmadi-hossein/leakage-audit)"
)

# ---------------------------------------------------------------------------
# Sidebar: data source and settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Dataset")
    source = st.radio(
        "Where do the images come from?",
        ["Demo dataset (built in)", "Upload a ZIP file", "Folder path on this machine"],
        key="source",
    )
    n_per_class = 40
    zip_file = None
    folder_path = ""
    if source == "Demo dataset (built in)":
        n_per_class = st.slider("Demo images per class", 10, 80, 40, step=10)
        st.caption(
            "Synthetic two-class data with planted exact copies and "
            "recompressed or resized near duplicates."
        )
    elif source == "Upload a ZIP file":
        zip_file = st.file_uploader(
            "ZIP with one subfolder per class", type=["zip"], key="zip_upload"
        )
        st.caption("Layout inside the ZIP: dataset_root/class_name/image.jpg")
    else:
        folder_path = st.text_input(
            "Folder path", placeholder="/path/to/dataset_root", key="folder_path"
        )
        st.caption("The folder must contain one subfolder per class.")

    st.header("Settings")
    threshold = st.slider(
        "pHash distance threshold (near duplicates)",
        min_value=0,
        max_value=12,
        value=5,
        help="Pairs of images whose perceptual hashes differ in at most "
        "this many bits are treated as near duplicates. 5 is the default "
        "used in the kidney-stone audit.",
        key="threshold",
    )
    seed = st.number_input("Random seed", value=42, step=1, key="seed")
    st.subheader("Split fractions")
    train_pct = st.slider("Train %", 50, 90, 70, step=5, key="train_pct")
    val_pct = st.slider("Validation %", 0, 40, 15, step=5, key="val_pct")
    test_pct = 100 - train_pct - val_pct
    st.write(f"Test %: **{test_pct}**")
    if test_pct <= 0:
        st.error("Train + validation must leave something for the test set.")

    run_audit = st.button(
        "Run audit", type="primary", disabled=test_pct <= 0, key="run_audit"
    )


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_demo(n: int) -> Path:
    """Generate the demo dataset once per (session, size) and return its root."""
    key = f"demo_root_{n}"
    if key not in st.session_state:
        root = Path(tempfile.mkdtemp(prefix="leakage_demo_")) / "demo_data"
        script = APP_DIR / "examples" / "make_demo_data.py"
        subprocess.run(
            [sys.executable, str(script), str(root), str(n)],
            check=True,
            capture_output=True,
            text=True,
        )
        st.session_state[key] = root
    return st.session_state[key]


def prepare_zip(uploaded) -> Path:
    """Extract an uploaded ZIP to a temp folder and find the dataset root."""
    raw = uploaded.getvalue()
    digest = str(len(raw)) + "_" + str(abs(hash(raw)) % 10**8)
    key = f"zip_root_{digest}"
    if key not in st.session_state:
        base = Path(tempfile.mkdtemp(prefix="leakage_zip_"))
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(base)
        root = base
        # If the ZIP wraps everything in a single top folder, step into it.
        entries = [p for p in base.iterdir() if not p.name.startswith("__")]
        if len(entries) == 1 and entries[0].is_dir():
            inner = entries[0]
            if any(d.is_dir() for d in inner.iterdir()):
                root = inner
        st.session_state[key] = root
    return st.session_state[key]


def resolve_root() -> Path | None:
    if source == "Demo dataset (built in)":
        return prepare_demo(int(n_per_class))
    if source == "Upload a ZIP file":
        if zip_file is None:
            st.info("Upload a ZIP file first, then press Run audit.")
            return None
        return prepare_zip(zip_file)
    if not folder_path.strip():
        st.info("Enter a folder path first, then press Run audit.")
        return None
    return Path(folder_path.strip())


# ---------------------------------------------------------------------------
# Run the audit
# ---------------------------------------------------------------------------
if run_audit:
    root = resolve_root()
    if root is not None:
        try:
            with st.spinner("Hashing every image and building duplicate clusters..."):
                dataset, cluster_of, summary = audit_dataset(
                    root, threshold=int(threshold)
                )
            st.session_state["audit"] = {
                "dataset": dataset,
                "cluster_of": cluster_of,
                "summary": summary,
                "threshold": int(threshold),
            }
            st.session_state.pop("split_csvs", None)
            st.session_state.pop("probe", None)
        except Exception as exc:  # show the real error, do not hide it
            st.session_state.pop("audit", None)
            st.error(f"Audit failed: {exc}")

audit = st.session_state.get("audit")

if audit is None:
    st.info(
        "Pick a dataset on the left and press **Run audit**. "
        "The demo dataset is the fastest way to see how it works: it has "
        "planted duplicates, so the audit should find them."
    )
else:
    dataset = audit["dataset"]
    cluster_of = audit["cluster_of"]
    s = audit["summary"]
    n = s["n_images"]

    st.header("1. Duplicate audit")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", f"{n:,}")
    c2.metric("Classes", len(s["class_counts"]))
    c3.metric("Exact duplicate groups", f"{s['exact_dup_groups']:,}")
    c4.metric(
        "Images that are exact copies", f"{s['exact_dup_images']:,}"
    )
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Near-duplicate pairs", f"{s['near_dup_pairs']:,}")
    c6.metric("Duplicate clusters", f"{s['dup_clusters']:,}")
    c7.metric(
        "Images inside duplicate clusters",
        f"{s['images_in_dup_clusters']:,} "
        f"({100 * s['images_in_dup_clusters_rate']:.1f}%)",
    )
    c8.metric("Clusters mixing classes", s["cross_class_clusters"])

    left, right = st.columns(2)
    with left:
        st.subheader("Images per class")
        counts_df = pd.DataFrame(
            {
                "class": list(s["class_counts"].keys()),
                "images": list(s["class_counts"].values()),
            }
        )
        st.bar_chart(counts_df, x="class", y="images")
    with right:
        st.subheader("Duplicate cluster sizes")
        sizes = Counter(
            int(v) for v in cluster_of.tolist()
        )
        multi_sizes = [sz for sz in sizes.values() if sz > 1]
        if multi_sizes:
            dist = Counter(multi_sizes)
            dist_df = pd.DataFrame(
                {
                    "cluster size": sorted(dist.keys()),
                    "number of clusters": [dist[k] for k in sorted(dist.keys())],
                }
            )
            st.bar_chart(dist_df, x="cluster size", y="number of clusters")
            st.caption(
                f"Largest cluster: {max(multi_sizes)} images. "
                "Every image in one cluster must stay on the same side "
                "of any train/test split."
            )
        else:
            st.write("No duplicate clusters found at this threshold.")

    # Show the largest cluster so the duplicates can be eyeballed.
    if multi_sizes:
        st.subheader("Largest duplicate cluster (check it with your eyes)")
        counts = Counter(int(v) for v in cluster_of.tolist())
        biggest_id = max(counts, key=lambda k: counts[k])
        members = np.where(cluster_of == biggest_id)[0][:8]
        cols = st.columns(len(members))
        for col, idx in zip(cols, members):
            path = dataset.files[int(idx)]
            with col:
                st.image(str(path), width="stretch")
                st.caption(
                    f"{dataset.class_names[dataset.labels[int(idx)]]} / {path.name}"
                )

    # ------------------------------------------------------------------
    st.header("2. What a normal random split does")
    fracs = (
        train_pct / 100.0,
        val_pct / 100.0,
        test_pct / 100.0,
    )
    with st.spinner("Measuring leakage of a random stratified split..."):
        leak = check_random_split(
            dataset,
            cluster_of,
            seed=int(seed),
            train_frac=fracs[0],
            val_frac=fracs[1],
            test_frac=fracs[2],
        )
    l1, l2, l3 = st.columns(3)
    l1.metric("Test images", f"{leak['test_size']:,}")
    l2.metric(
        "Test images with a duplicate in train",
        f"{leak['test_cluster_leakage']:,} "
        f"({100 * leak['test_cluster_leakage_rate']:.1f}%)",
    )
    l3.metric("Test images that are exact copies of train images",
              f"{leak['test_exact_leakage']:,}")
    if leak["test_cluster_leakage_rate"] > 0:
        st.warning(
            "This is the leakage a standard random split hides: part of "
            "the test set was effectively already seen in training, so "
            "the test score overstates real performance."
        )
    else:
        st.success("No leakage under a random split at this threshold.")

    # ------------------------------------------------------------------
    st.header("3. Leakage-free split (clusters stay together)")
    if st.button("Build leakage-free split and CSVs", key="make_split"):
        train_idx, val_idx, test_idx = cluster_aware_split(
            dataset,
            cluster_of,
            seed=int(seed),
            train_frac=fracs[0],
            val_frac=fracs[1],
            test_frac=fracs[2],
        )
        check = measure_split_leakage(dataset, cluster_of, train_idx, test_idx)
        out_dir = Path(tempfile.mkdtemp(prefix="leakage_split_")) / "split"
        written = write_split_csvs(
            dataset,
            cluster_of,
            {"train": train_idx, "val": val_idx, "test": test_idx},
            out_dir,
        )
        st.session_state["split_csvs"] = {
            name: Path(path).read_bytes() for name, path in written.items()
        }
        st.session_state["split_check"] = {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "leakage": check["test_cluster_leakage"],
        }
    split_check = st.session_state.get("split_check")
    if split_check:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Train images", f"{split_check['train']:,}")
        s2.metric("Validation images", f"{split_check['val']:,}")
        s3.metric("Test images", f"{split_check['test']:,}")
        s4.metric("Test leakage", split_check["leakage"])
        st.success(
            "Leakage is zero by construction: no duplicate cluster "
            "crosses the train/test boundary."
        )
        csvs = st.session_state["split_csvs"]
        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "Download train.csv", csvs["train"], "train.csv", "text/csv"
        )
        d2.download_button(
            "Download val.csv", csvs["val"], "val.csv", "text/csv"
        )
        d3.download_button(
            "Download test.csv", csvs["test"], "test.csv", "text/csv"
        )

    # ------------------------------------------------------------------
    st.header("4. Acquisition-bias probe")
    st.markdown(
        "A logistic regression that sees **only 11 crude statistics** "
        "per image (colour means, border and centre brightness, file "
        "size, dimensions) and never sees the content. If it predicts "
        "the class well above chance, the classes were collected "
        "differently and the dataset itself carries class signal."
    )
    if st.button("Run probe", key="run_probe"):
        with st.spinner("Computing image statistics and fitting the probe..."):
            st.session_state["probe"] = run_probe(dataset, seed=int(seed))
    probe = st.session_state.get("probe")
    if probe:
        p1, p2, p3 = st.columns(3)
        p1.metric("Probe accuracy", f"{100 * probe['probe_accuracy']:.1f}%")
        p2.metric(
            "Always-guess-majority baseline",
            f"{100 * probe['majority_class_rate']:.1f}%",
        )
        p3.metric("Chance level", f"{100 * probe['chance_level']:.1f}%")
        st.caption(f"Method: {probe['method']}. Features: {', '.join(FEATURE_NAMES)}.")
        if probe["probe_accuracy"] > probe["majority_class_rate"] + 0.05:
            st.warning(
                "The probe beats the majority baseline by more than 5 "
                "points: acquisition bias is present. Even a leakage-free "
                "split can overstate how the model would do on data from "
                "another device or site."
            )
        else:
            st.success(
                "The probe stays close to the majority baseline: no "
                "strong acquisition bias detected by these statistics."
            )

st.divider()
st.caption(
    "Case study the tool was built from: a public ultrasound kidney-stone "
    "dataset of 9,416 images, where 85.8% of a random split's test images "
    "had a duplicate in training, yet accuracy stayed at 99.72% on a "
    "leakage-free split, and this probe reached 84.8% from image "
    "statistics alone. Paper: IICAI2027, Paper ID 1004 (under review)."
)
