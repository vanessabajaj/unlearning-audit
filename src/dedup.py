"""Group near-duplicate frames so data splits can be made by group, not by image.

Kvasir-SEG has no patient or video IDs, and consecutive frames of one polyp can appear as
separate images. Splitting by image would leak the same polyp into train and test.

Rule (link two images if either holds):
  strict: perceptual-hash Hamming distance <= hash_strict
  loose : Hamming distance <= hash_loose AND shift-tolerant grayscale SSIM (64x64) >= ssim_min
          (max SSIM over translations up to +/-shift_px, so consecutive video frames that
          moved slightly still match; plain SSIM misses them)
Linked images are merged transitively (union-find); each connected component is a group.

Usage:
  python src/dedup.py --root data/raw/Kvasir-SEG --out data/groups
Outputs: groups.csv, pairs.csv (all candidate pairs with scores), threshold_sweep.csv,
         review_linked.png / review_borderline.png (for visual threshold checks).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import uniform_filter

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Thresholds:
    hash_strict: int = 4
    hash_loose: int = 24
    ssim_min: float = 0.75
    shift_px: int = 5     # at 64x64, about +/-8% of the frame
    shift_step: int = 1


def list_pairs(root: Path) -> pd.DataFrame:
    """Match images/<stem>.* with masks/<stem>.*; fail loudly on any mismatch."""
    imgs = {p.stem: p for p in (root / "images").iterdir() if p.suffix.lower() in IMG_EXT}
    masks = {p.stem: p for p in (root / "masks").iterdir() if p.suffix.lower() in IMG_EXT}
    if set(imgs) != set(masks):
        raise ValueError(f"image/mask mismatch: {len(set(imgs) ^ set(masks))} unpaired stems, "
                         f"e.g. {sorted(set(imgs) ^ set(masks))[:5]}")
    ids = sorted(imgs)
    return pd.DataFrame({"image_id": ids, "image_path": [str(imgs[i]) for i in ids],
                         "mask_path": [str(masks[i]) for i in ids]})


def gray_thumb(path: str, size: int) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float64)


def phash_bits(gray32: np.ndarray) -> np.ndarray:
    """64-bit DCT perceptual hash (8x8 low frequencies, DC excluded from the median)."""
    low = dctn(gray32, norm="ortho")[:8, :8].ravel()
    return low > np.median(low[1:])


def ssim(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    """Mean SSIM with a uniform window (Wang et al. 2004 constants, data range 255)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = uniform_filter(a, win), uniform_filter(b, win)
    va = uniform_filter(a * a, win) - mu_a**2
    vb = uniform_filter(b * b, win) - mu_b**2
    cov = uniform_filter(a * b, win) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
    return float(s.mean())


def shift_ssim(a: np.ndarray, b: np.ndarray, max_shift: int, step: int = 1) -> float:
    """Best SSIM over integer translations of b relative to a (overlapping region only)."""
    n = a.shape[0]
    best = -1.0
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            A = a[max(0, dy):n + min(0, dy), max(0, dx):n + min(0, dx)]
            B = b[max(0, -dy):n + min(0, -dy), max(0, -dx):n + min(0, -dx)]
            best = max(best, ssim(A, B))
    return best


def _shift_ssim_job(job):
    return shift_ssim(*job)


def mask_area_frac(path: str) -> float:
    with Image.open(path) as im:
        m = np.asarray(im.convert("L")) > 127
    return float(m.mean())


def compute_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    bits = np.stack([phash_bits(gray_thumb(p, 32)) for p in df.image_path])
    thumbs = np.stack([gray_thumb(p, 64) for p in df.image_path])
    return bits, thumbs


def candidate_pairs(bits: np.ndarray, thumbs: np.ndarray, t: "Thresholds") -> pd.DataFrame:
    b = bits.astype(np.uint8)
    ham = (b[:, None, :] != b[None, :, :]).sum(-1)       # n x n, fine for a few thousand images
    i, j = np.where(np.triu(ham <= t.hash_loose, k=1))
    jobs = [(thumbs[a], thumbs[c], t.shift_px, t.shift_step) for a, c in zip(i, j)]
    if len(jobs) > 200:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as ex:  # fork is unsafe with threads
            s = list(ex.map(_shift_ssim_job, jobs, chunksize=64))
    else:
        s = [_shift_ssim_job(job) for job in jobs]
    return pd.DataFrame({"i": i, "j": j, "hamming": ham[i, j], "ssim": s})


def link(pairs: pd.DataFrame, t: Thresholds) -> np.ndarray:
    return ((pairs.hamming <= t.hash_strict)
            | ((pairs.hamming <= t.hash_loose) & (pairs.ssim >= t.ssim_min))).to_numpy()


def connected_groups(n: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, c in edges:
        ra, rc = find(a), find(c)
        if ra != rc:
            parent[max(ra, rc)] = min(ra, rc)
    roots = np.array([find(x) for x in range(n)])
    _, gid = np.unique(roots, return_inverse=True)  # dense ids ordered by smallest member
    return gid


def group_images(df: pd.DataFrame, t: Thresholds = Thresholds()):
    bits, thumbs = compute_features(df)
    pairs = candidate_pairs(bits, thumbs, t)
    pairs["linked"] = link(pairs, t)
    gid = connected_groups(len(df), pairs.loc[pairs.linked, ["i", "j"]].to_numpy())
    out = df.copy()
    out["group_id"] = gid
    out["group_size"] = out.groupby("group_id").image_id.transform("size")
    out["mask_area_frac"] = [mask_area_frac(p) for p in df.mask_path]
    pairs["image_i"] = df.image_id.to_numpy()[pairs.i]
    pairs["image_j"] = df.image_id.to_numpy()[pairs.j]
    return out, pairs


def threshold_sweep(n: int, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hs in (2, 4, 6, 8):
        for sm in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            t = Thresholds(hash_strict=hs, ssim_min=sm)
            g = connected_groups(n, pairs.loc[link(pairs, t), ["i", "j"]].to_numpy())
            sizes = np.bincount(g)
            rows.append(dict(hash_strict=hs, ssim_min=sm, n_groups=len(sizes),
                             images_in_multi=int(sizes[sizes > 1].sum()), largest_group=int(sizes.max())))
    return pd.DataFrame(rows)


def review_montage(df, pairs, path, n=24, thumb=128):
    """Side-by-side pairs for a human to judge; one row per pair, scores in the filename-free caption."""
    from PIL import ImageDraw
    pairs = pairs.head(n)
    if pairs.empty:
        return
    canvas = Image.new("RGB", (2 * thumb + 170, thumb * len(pairs)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(pairs.itertuples()):
        for k, idx in enumerate((row.i, row.j)):
            with Image.open(df.image_path.iloc[idx]) as im:
                canvas.paste(im.convert("RGB").resize((thumb, thumb)), (k * thumb, r * thumb))
        draw.text((2 * thumb + 6, r * thumb + 6),
                  f"ham={row.hamming}\nssim={row.ssim:.2f}\n{row.image_i[:14]}\n{row.image_j[:14]}", fill="black")
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hash-strict", type=int, default=Thresholds.hash_strict)
    ap.add_argument("--hash-loose", type=int, default=Thresholds.hash_loose)
    ap.add_argument("--ssim-min", type=float, default=Thresholds.ssim_min)
    ap.add_argument("--shift-px", type=int, default=Thresholds.shift_px)
    a = ap.parse_args()
    t = Thresholds(a.hash_strict, a.hash_loose, a.ssim_min, a.shift_px)
    a.out.mkdir(parents=True, exist_ok=True)
    df = list_pairs(a.root)
    groups, pairs = group_images(df, t)
    groups.to_csv(a.out / "groups.csv", index=False)
    pairs.to_csv(a.out / "pairs.csv", index=False)
    threshold_sweep(len(df), pairs).to_csv(a.out / "threshold_sweep.csv", index=False)
    linked = pairs[pairs.linked].sort_values("ssim")              # weakest links first
    border = pairs[~pairs.linked].sort_values("ssim", ascending=False)  # closest misses first
    review_montage(df, linked, a.out / "review_linked.png")
    review_montage(df, border, a.out / "review_borderline.png")
    sizes = groups.drop_duplicates("group_id").group_size
    print(f"images={len(df)} groups={len(sizes)} multi-image groups={int((sizes > 1).sum())} "
          f"images in them={int(sizes[sizes > 1].sum())} largest={int(sizes.max())}")
    print(f"thresholds: {t}")


if __name__ == "__main__":
    main()
