"""Pilot step 2: does a fine-tuned MedSAM mask decoder remember its training images at all?
And does a varied-box audit see that better than the single tight box?

Design (two-model swap, so image difficulty cancels):
  split the cached images into halves A and B; fine-tune decoder copy on A -> model_A, on B -> model_B
  (same start, recipe and seed). For every image i:
      d_i = score(model that trained on i, i) - score(model that did not, i)
  A real memory effect makes d_i > 0 on average. Compare d for the tight box vs K varied boxes.

Limits: decoder-only fine-tuning (no GPU here), one seed, one split, Kvasir-SEG images (possibly in
MedSAM's own pretraining). It answers "is there any signal", not "how big is it in the full study".
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score
from transformers import SamModel

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pilot" / "cache"
OUT = ROOT / "pilot"


def forward(model, emb, boxes):
    """emb (B,256,64,64) float, boxes (B,4) in 1024 coords -> logits (B,256,256)."""
    out = model(image_embeddings=emb, input_boxes=boxes[:, None, :], multimask_output=False)
    return out.pred_masks[:, 0, 0]


def jitter(box, rng, max_px=20.0):
    """Training jitter: each side moves independently by up to +-max_px (1024 scale)."""
    b = box + rng.uniform(-max_px, max_px, size=4).astype(np.float32)
    b = np.clip(b, 0, 1024)
    return np.array([min(b[0], b[2] - 4), min(b[1], b[3] - 4), max(b[2], b[0] + 4), max(b[3], b[1] + 4)],
                    dtype=np.float32)


def seg_loss(logits, gt):
    bce = F.binary_cross_entropy_with_logits(logits, gt)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * gt).sum((1, 2)) + 1) / (p.sum((1, 2)) + gt.sum((1, 2)) + 1)
    return bce + dice.mean()


def finetune(base, emb, gt, box, epochs, lr, seed, bs=8):
    model = copy.deepcopy(base)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = epochs * int(np.ceil(len(emb) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.train()
    for ep in range(epochs):
        order = rng.permutation(len(emb))
        tot = 0.0
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            b = torch.from_numpy(np.stack([jitter(box[i].numpy(), rng) for i in idx]))
            loss = seg_loss(forward(model, emb[idx].float(), b), gt[idx].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        if ep in (0, epochs - 1) or (ep + 1) % 10 == 0:
            print(f"  epoch {ep + 1}/{epochs} loss {tot / len(emb):.4f}", flush=True)
    return model.eval()


def varied_boxes(box, gt, k, rng, max_shift=0.3, scale=(0.8, 1.5), aspect=(0.8, 1.25), min_cover=0.7):
    """K plausible boxes: shifted / rescaled, must still cover >= min_cover of the lesion."""
    x0, y0, x1, y1 = box
    w, h, cx, cy = x1 - x0, y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2
    lesion = gt.astype(bool)
    total = lesion.sum()
    out = []
    while len(out) < k:
        s = rng.uniform(*scale)
        a = rng.uniform(*aspect)
        nw, nh = w * s * np.sqrt(a), h * s / np.sqrt(a)
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0, max_shift)
        ncx, ncy = cx + r * w * np.cos(ang), cy + r * h * np.sin(ang)
        b = np.clip([ncx - nw / 2, ncy - nh / 2, ncx + nw / 2, ncy + nh / 2], 0, 1024).astype(np.float32)
        bx = (b / 4).round().astype(int)                 # 256-grid coverage check
        cover = lesion[bx[1]:bx[3], bx[0]:bx[2]].sum() / max(total, 1)
        if cover >= min_cover:
            out.append(b)
    return np.stack(out)


def frame_stats(logits, gt):
    """logits (Q,256,256), gt (256,256) -> per-query margin and Dice."""
    g = gt.bool()
    margin = logits[:, g].mean(1) - logits[:, ~g].mean(1)
    pred = logits > 0
    inter = (pred & g).sum((1, 2)).float()
    dice = 2 * inter / (pred.sum((1, 2)) + g.sum() + 1e-6)
    return margin.numpy(), dice.numpy(), pred


def stability(pred):
    q = pred.shape[0]
    ious = []
    for i in range(q):
        for j in range(i + 1, q):
            u = (pred[i] | pred[j]).sum().item()
            ious.append((pred[i] & pred[j]).sum().item() / u if u else 1.0)
    return float(np.mean(ious)) if ious else np.nan


@torch.no_grad()
def audit(model, emb, gt, box, k, seed):
    res = []
    for i in range(len(emb)):
        rng = np.random.default_rng(seed + i)                 # same boxes for every model
        vb = varied_boxes(box[i].numpy(), gt[i].numpy(), k, rng)
        boxes = torch.from_numpy(np.concatenate([box[i].numpy()[None], vb]))
        e = emb[i:i + 1].float().expand(len(boxes), -1, -1, -1)
        logits = forward(model, e, boxes)
        m, d, pred = frame_stats(logits, gt[i])
        res.append(dict(tight_margin=m[0], tight_dice=d[0], box_margin=m[1:].mean(), box_dice=d[1:].mean(),
                        box_stability=stability(pred[1:])))
    return {key: np.array([r[key] for r in res]) for key in res[0]}


def summarise(d):
    t = stats.ttest_1samp(d, 0.0)
    return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), dz=float(d.mean() / d.std(ddof=1)),
                frac_pos=float((d > 0).mean()), t=float(t.statistic), p=float(t.pvalue))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="e40")
    ap.add_argument("--cache", default="embeddings.pt")
    args = ap.parse_args()
    torch.set_num_threads(4)

    c = torch.load(CACHE / args.cache)
    n = len(c["emb"])
    perm = np.random.default_rng(1000 + args.seed).permutation(n)
    A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    base = SamModel.from_pretrained(c["model_id"])

    t0 = time.time()
    models = {}
    for name, idx in (("A", A), ("B", B)):
        print(f"fine-tune model_{name} on {len(idx)} images", flush=True)
        models[name] = finetune(base, c["emb"][idx], c["gt"][idx], c["box"][idx], args.epochs, args.lr, args.seed)
    print(f"fine-tuning took {time.time() - t0:.0f} s", flush=True)

    sc = {m: audit(models[m], c["emb"], c["gt"], c["box"], args.k, seed=777) for m in models}
    sc["base"] = audit(base.eval(), c["emb"], c["gt"], c["box"], args.k, seed=777)
    member = np.zeros(n, dtype=bool)
    member[A] = True                                  # True = image is in model_A's training half

    report = dict(n_images=n, k=args.k, epochs=args.epochs, lr=args.lr, seed=args.seed,
                  fine_tune="mask decoder only (CPU)", data="Kvasir-SEG singletons")
    # Dice of each model on its own half vs the other half (ordinary generalisation gap)
    report["dice_tight"] = {
        "base_all": float(sc["base"]["tight_dice"].mean()),
        "own_half": float(np.r_[sc["A"]["tight_dice"][A], sc["B"]["tight_dice"][B]].mean()),
        "other_half": float(np.r_[sc["A"]["tight_dice"][B], sc["B"]["tight_dice"][A]].mean()),
    }
    report["swap_test"] = {}
    for key in ["tight_margin", "box_margin", "tight_dice", "box_dice", "box_stability"]:
        d = np.where(member, sc["A"][key] - sc["B"][key], sc["B"][key] - sc["A"][key])
        # membership AUC for model_A with difficulty calibrated by model_B: members should score higher
        auc = roc_auc_score(member, sc["A"][key] - sc["B"][key])
        report["swap_test"][key] = {**summarise(d), "auc": float(auc)}
    # paired: is the per-image memory effect bigger for varied boxes than for the tight box?
    zt = lambda key: np.where(member, sc["A"][key] - sc["B"][key], sc["B"][key] - sc["A"][key])
    for stat in ["margin", "dice"]:
        dt, db = zt(f"tight_{stat}"), zt(f"box_{stat}")
        # standardise each by its own SD so the comparison is in signal-to-noise units
        report[f"box_vs_tight_{stat}"] = dict(dz_tight=float(dt.mean() / dt.std(ddof=1)),
                                             dz_box=float(db.mean() / db.std(ddof=1)))
    (OUT / f"results_{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
