#!/usr/bin/env python3
"""Generate a small synthetic dataset with planted duplicates.

Creates two classes (alpha, beta) of simple synthetic images. Some images
are copied exactly, and others are saved again after a resize or a JPEG
recompression so that they are near duplicates rather than byte-identical.

Usage:
    python examples/make_demo_data.py [output_dir] [n_per_class]

Then try:
    leakage-audit audit demo_data
    leakage-audit make-split demo_data
    leakage-audit probe demo_data
"""
import io
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def make_image(rng: np.random.Generator, base_color, seed_offset: int) -> Image.Image:
    """Draw a simple synthetic 'scan': gradient background plus blobs."""
    size = 128
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size, 3), dtype=np.float64)
    for c in range(3):
        img[:, :, c] = base_color[c] + 30 * (xx / size) + 10 * np.sin(yy / 9.0)
    local = np.random.default_rng(seed_offset)
    for _ in range(6):
        cx, cy = local.integers(15, size - 15, size=2)
        r = local.integers(6, 18)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        img[mask] = img[mask] * 0.4 + np.array(base_color) * 0.6 + 40
    noise = rng.normal(0, 4, img.shape)
    return Image.fromarray(np.clip(img + noise, 0, 255).astype(np.uint8))


def recompress(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=55)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_data")
    n_per_class = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rng = np.random.default_rng(7)
    classes = {"alpha": (90, 120, 160), "beta": (170, 110, 80)}
    planted_exact = 0
    planted_near = 0
    for cls, color in classes.items():
        folder = out / cls
        folder.mkdir(parents=True, exist_ok=True)
        originals: list[Path] = []
        for i in range(n_per_class):
            img = make_image(rng, color, seed_offset=hash((cls, i)) % 100000)
            path = folder / f"{cls}_{i:03d}.jpg"
            img.save(path, quality=92)
            originals.append(path)
        # Plant exact duplicates of the first 8 originals.
        for src in originals[:8]:
            dst = folder / (src.stem + "_copy.jpg")
            shutil.copyfile(src, dst)
            planted_exact += 1
        # Plant near duplicates of the next 8: recompressed or resized.
        for k, src in enumerate(originals[8:16]):
            img = Image.open(src)
            if k % 2 == 0:
                near = recompress(img)
            else:
                near = img.resize((112, 112)).resize((128, 128))
            near.save(folder / (src.stem + "_near.jpg"), quality=92)
            planted_near += 1
    total = sum(1 for _ in out.rglob("*.jpg"))
    print(f"Wrote {total} images to {out}")
    print(f"Planted exact duplicates: {planted_exact}")
    print(f"Planted near duplicates: {planted_near}")


if __name__ == "__main__":
    main()
