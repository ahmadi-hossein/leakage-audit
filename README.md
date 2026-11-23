# leakage-audit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A two-stage audit for image classification datasets. It finds the duplicate
images that quietly leak across train/test splits, builds a split that does
not leak, and probes whether the dataset itself gives away the class through
acquisition artefacts.

## Why it matters

A test set only measures generalisation if it is independent of the training
set. Public image datasets routinely contain exact copies and near duplicates
of the same underlying image, and a random split scatters them across train
and test. The model is then partly tested on images it has already seen, and
the reported accuracy is too optimistic. Leakage of this kind is a recurring
cause of irreproducible results in machine-learning-based science
(Kapoor and Narayanan, 2023), and the problem of data leakage in data mining
was formalised by Kaufman et al. (2012).

Duplicates are only half of the story. If the classes were collected with
different devices, compression settings or framing, crude image statistics
carry class signal on their own. A model can score well by learning the
acquisition pipeline instead of the content, and even a perfectly clean split
will not reveal that. The audit therefore has two stages:

1. **Duplicate-leakage audit.** Exact duplicates by MD5 checksum, near
   duplicates by perceptual hash (pHash), merged into clusters with
   union-find. The audit reports how much of a standard random split leaks.
2. **Acquisition-bias probe.** A logistic regression trained on 11 crude
   image statistics only (channel means, intensity spread, border and centre
   statistics, file size, dimensions). Accuracy well above the majority
   baseline means the dataset leaks class information through acquisition.

## Install

```bash
pip install .
```

Requires Python 3.10+ and depends on numpy, Pillow, ImageHash and
scikit-learn.

## Web app (Streamlit)

Prefer clicking to a command line? The repo ships a Streamlit app
(`app.py`) with the same audit: pick the built-in demo dataset or upload
a ZIP of your own images (one subfolder per class), run the duplicate
audit, see how much a random split leaks, download a leakage-free
cluster-aware split, and run the acquisition-bias probe.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The same app deploys free on Streamlit Community Cloud: point it at
this repository with `app.py` as the main file.

## Quickstart

The dataset layout is one subfolder per class:

```text
data/
  Normal/
    img001.jpg
    ...
  Stone/
    img001.jpg
    ...
```

```bash
# Full audit: duplicates, clusters, random-split leakage, Markdown report
leakage-audit audit data --out audit_summary.json --report audit_report.md

# Only measure how much a random stratified split leaks
leakage-audit check-split data

# Write a leakage-free cluster-aware split (train/val/test CSVs)
leakage-audit make-split data --out-dir splits

# Run the acquisition-bias probe
leakage-audit probe data
```

Every command prints a readable report and writes a JSON summary.

There is also a demo dataset generator with planted duplicates:

```bash
python examples/make_demo_data.py demo_data
leakage-audit audit demo_data
```

## Case study: a public kidney-stone ultrasound dataset

The method comes from an audit study of a public ultrasound kidney-stone
dataset (9,416 images, classes Normal and Stone). The audit found:

- 907 exact-duplicate MD5 groups covering 1,947 images.
- 2,737 duplicate clusters (exact plus pHash Hamming distance <= 5)
  covering 9,366 of the 9,416 images.
- Under a random stratified 70/15/15 split (seed 42), 85.8 percent of test
  images had a duplicate-cluster mate in the training set, and 221 test
  images were exact copies of training images.
- A logistic regression on the 11 crude image statistics alone reached
  82.9 percent accuracy, far above chance, showing that the classes also
  differ in acquisition characteristics.

A model that scores near 100 percent on the random split is therefore not
evidence of clinical readiness: much of the test set was effectively seen
during training, and part of the remaining signal is acquisition bias rather
than anatomy.

## How it works

- **Exact duplicates.** MD5 of the raw file bytes. Byte-identical files are
  grouped directly.
- **Near duplicates.** Perceptual hash (pHash, hash_size 8) per image.
  Pairs with Hamming distance at most 5 (configurable with `--threshold`)
  are treated as near duplicates. Pairwise distances are computed in chunks
  so memory stays bounded on large datasets.
- **Clustering.** Exact groups and near-duplicate pairs are merged with
  union-find. A cluster is the smallest unit that a split may move; if any
  member of a cluster is in train, no member may be in test.
- **Leakage check.** A random stratified split is reproduced and the audit
  counts test images whose cluster, or exact checksum, appears in train.
- **Cluster-aware split.** Group shuffling over cluster ids produces
  train/val/test CSVs with zero cluster leakage by construction.
- **Probe.** The 11 statistics are standardised and fed to a logistic
  regression, evaluated with stratified cross-validation (or a holdout for
  very small datasets).

## Limitations

- The dataset used in the case study has no patient identifiers, so no
  patient-level split is possible. If your data has subject identifiers,
  group by those first; duplicate clusters are a fallback, not a substitute.
- The pHash threshold is a trade-off. A lower threshold misses recompressed
  or cropped copies; a higher one merges genuinely different images that
  happen to look similar. Inspect the largest clusters before trusting a
  threshold on a new dataset.
- A clean split removes duplicate leakage, not acquisition bias. The probe
  flags the bias; only validation on data from a different source can rule
  it out.
- Perceptual hashing catches pixel-level similarity. Duplicates that were
  heavily cropped, rotated or re-acquired from the same subject will not
  cluster, which is another reason subject-level grouping matters when
  identifiers exist.

## References

- Kapoor, S. and Narayanan, A. (2023). Leakage and the reproducibility
  crisis in machine-learning-based science. Patterns 4(9), 100804.
- Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012). Leakage in
  data mining: Formulation, detection, and avoidance. ACM Transactions on
  Knowledge Discovery from Data 6(4), Article 15.

## License

MIT. See [LICENSE](LICENSE).

## Author

Hossein Ahmadi - [github.com/ahmadi-hossein](https://github.com/ahmadi-hossein)
