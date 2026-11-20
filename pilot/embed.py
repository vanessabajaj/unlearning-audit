"""Pilot step 1: cache MedSAM image embeddings for a subset of Kvasir-SEG (CPU).

Only singleton images (no near-duplicates, group_size == 1) are used, so no image has a copy on
the other side of the member/non-member split. Output: pilot/cache/embeddings.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pilot" / "cache"
MODEL_ID = "wanglab/medsam-vit-base"


def gt_256(mask_path, scale):
    """Ground-truth mask on the decoder's 256x256 grid (resize longest side to 1024, pad, /4)."""
    m = Image.open(mask_path).convert("L")
    w, h = m.size
    nw, nh = int(round(w * scale)), int(round(h * scale))
    m = m.resize((nw, nh), Image.NEAREST)
    canvas = Image.new("L", (1024, 1024), 0)
    canvas.paste(m, (0, 0))
    small = np.asarray(canvas.resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
    return (small > 0.5).astype(np.uint8)


def tight_box(mask_path, scale):
    m = np.asarray(Image.open(mask_path).convert("L")) > 127
    ys, xs = np.nonzero(m)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()
    torch.set_num_threads(4)
    CACHE.mkdir(parents=True, exist_ok=True)

    g = pd.read_csv(ROOT / "data" / "groups" / "groups.csv")
    g = g[g.group_size == 1].sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    proc = SamProcessor.from_pretrained(MODEL_ID)
    model = SamModel.from_pretrained(MODEL_ID).eval()

    embs, gts, boxes, sizes = [], [], [], []
    t0 = time.time()
    for i, r in g.iterrows():
        img = Image.open(ROOT / r.image_path).convert("RGB")
        w, h = img.size
        scale = 1024 / max(w, h)
        inp = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            e = model.get_image_embeddings(inp["pixel_values"])
        embs.append(e[0].half())
        gts.append(torch.from_numpy(gt_256(ROOT / r.mask_path, scale)))
        boxes.append(torch.from_numpy(tight_box(ROOT / r.mask_path, scale)))
        sizes.append((int(round(w * scale)), int(round(h * scale))))
        if i in (0, 4) or (i + 1) % 40 == 0:
            print(f"{i + 1}/{len(g)} images, {(time.time() - t0) / (i + 1):.1f} s/image", flush=True)
    torch.save(dict(image_id=list(g.image_id), emb=torch.stack(embs), gt=torch.stack(gts),
                    box=torch.stack(boxes), size=sizes, model_id=MODEL_ID), CACHE / "embeddings.pt")
    print("saved", CACHE / "embeddings.pt")


if __name__ == "__main__":
    main()
