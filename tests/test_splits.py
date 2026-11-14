"""Synthetic check of near-duplicate grouping and group-wise splitting.

Builds a fake Kvasir-SEG layout: 80 distinct 'polyp' scenes, each with 0-3 near-duplicate
frames (small shift/zoom, brightness change, JPEG re-encode), at varied image sizes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dedup import group_images, list_pairs  # noqa: E402
from make_splits import TARGETS, assign, check  # noqa: E402


def scene(rng, w, h):
    low = rng.normal(size=(6, 8, 3))
    field = np.asarray(Image.fromarray(((low - low.min()) / np.ptp(low) * 255).astype(np.uint8))
                       .resize((w, h), Image.BICUBIC)).copy()
    mask = Image.new("L", (w, h), 0)
    cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
    r = rng.uniform(0.05, 0.3) * min(w, h)
    ImageDraw.Draw(mask).ellipse([cx - r, cy - r * 0.8, cx + r, cy + r * 0.8], fill=255)
    m = np.asarray(mask) > 0
    field[m] = (0.5 * field[m] + 0.5 * np.array([220, 120, 110])).astype(np.uint8)
    return Image.fromarray(field), mask


def jitter(rng, img, mask):
    w, h = img.size
    dx, dy = (rng.uniform(-0.04, 0.04, 2) * (w, h)).astype(int)
    z = rng.uniform(0.0, 0.05)
    box = (max(0, int(z * w) + dx), max(0, int(z * h) + dy),
           min(w, int((1 - z) * w) + dx), min(h, int((1 - z) * h) + dy))
    img2 = ImageEnhance.Brightness(img.crop(box).resize((w, h))).enhance(rng.uniform(0.9, 1.1))
    return img2, mask.crop(box).resize((w, h))


@pytest.fixture(scope="module")
def fake_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("kvasir")
    (root / "images").mkdir()
    (root / "masks").mkdir()
    rng = np.random.default_rng(7)
    truth = {}
    for s in range(80):
        w, h = int(rng.integers(330, 640)), int(rng.integers(300, 540))
        img, mask = scene(rng, w, h)
        for k in range(1 + int(rng.integers(0, 4))):
            im, mk = (img, mask) if k == 0 else jitter(rng, img, mask)
            name = f"scene{s:03d}_f{k}"
            im.save(root / "images" / f"{name}.jpg", quality=90)
            mk.save(root / "masks" / f"{name}.jpg", quality=95)
            truth[name] = s
    return root, truth


@pytest.fixture(scope="module")
def grouped(fake_dataset):
    root, _ = fake_dataset
    groups, _ = group_images(list_pairs(root))
    return groups


def test_grouping_recovers_scenes(fake_dataset, grouped):
    _, truth = fake_dataset
    groups = grouped
    true_scene = groups.image_id.map(truth)
    # every scene's frames land in one group (recall) ...
    assert (groups.groupby(true_scene).group_id.nunique() == 1).all()
    # ... and no group mixes different scenes (precision)
    assert (groups.groupby("group_id").image_id.apply(lambda s: s.map(truth).nunique()) == 1).all()
    assert groups.group_id.nunique() == 80


def test_splits_keep_groups_whole_and_hit_targets(grouped):
    df = assign(grouped, seed=1)
    summary = check(df)  # raises if any group spans splits
    for s, f in TARGETS.items():
        assert abs(summary[s]["frac"] - f) < 0.05, (s, summary[s])
    # every partition gets all three polyp-size terciles
    for s in TARGETS:
        assert min(summary[s]["terciles"].values()) > 0, (s, summary[s])


def test_splits_reproducible_and_seed_sensitive(grouped):
    a, b, c = assign(grouped, seed=3), assign(grouped, seed=3), assign(grouped, seed=4)
    assert a.split.equals(b.split)
    assert not a.split.equals(c.split)


def test_unpaired_mask_fails_loudly(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "a.jpg")
    with pytest.raises(ValueError, match="unpaired"):
        list_pairs(tmp_path)


def test_check_catches_leak():
    df = pd.DataFrame({"group_id": [0, 0, 1], "split": ["train_pool", "test", "val"],
                       "size_tercile": ["small", "small", "large"]})
    with pytest.raises(AssertionError, match="span"):
        check(df)
