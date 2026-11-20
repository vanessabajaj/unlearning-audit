"""Pilot step 3: after unlearning, does a trace of the withdrawn images remain, and do varied boxes
see it better than the tight box?

Split of the 240 cached Kvasir-SEG singletons (one seed):
  W (withdrawn) 40 | retain 120 | unseen 80
Models (mask decoder only, CPU):
  full       fine-tuned on W + retain
  oracle     fine-tuned on retain only (same recipe and seed)  -> what perfect unlearning looks like
  retain-ft  full, then 5 more epochs on retain only
  neggrad+   full, then 5 epochs of: loss on retain  minus  0.5 x loss on W (W loss capped)
Audit statistic per image: d_i = score(model, i) - score(oracle, i).
Trace = mean d on W minus mean d on unseen images (unseen images were never trained on by either
model, so they absorb any general difference between the two models). AUC = how well d separates
W images from unseen ones (0.5 = no trace).
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from transformers import SamModel

from membership_pilot import audit, finetune, forward, jitter, seg_loss

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pilot" / "cache"
OUT = ROOT / "pilot"


def neggrad_plus(start, emb, gt, box, w_idx, r_idx, epochs, lr, seed, lam=0.5, cap=1.0, bs=8):
    model = copy.deepcopy(start)
    for p in model.mask_decoder.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.mask_decoder.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(epochs):
        ro = rng.permutation(r_idx)
        for s in range(0, len(ro), bs):
            ri = ro[s:s + bs]
            wi = rng.choice(w_idx, size=min(bs, len(w_idx)), replace=False)
            rb = torch.from_numpy(np.stack([jitter(box[i].numpy(), rng) for i in ri]))
            wb = torch.from_numpy(np.stack([jitter(box[i].numpy(), rng) for i in wi]))
            lr_ = seg_loss(forward(model, emb[ri].float(), rb), gt[ri].float())
            lw = seg_loss(forward(model, emb[wi].float(), wb), gt[wi].float())
            loss = lr_ - lam * torch.clamp(lw, max=cap)      # cap stops the ascent from blowing up
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  neggrad+ epoch {ep + 1}/{epochs} retain {lr_.item():.3f} withdrawn {lw.item():.3f}", flush=True)
    return model.eval()


def trace(model_sc, oracle_sc, key, W, U):
    d = model_sc[key] - oracle_sc[key]
    dw, du = d[W], d[U]
    t = stats.ttest_ind(dw, du, equal_var=False)
    lab = np.r_[np.ones(len(W)), np.zeros(len(U))]
    pooled = np.sqrt((dw.var(ddof=1) + du.var(ddof=1)) / 2)
    return dict(trace=float(dw.mean() - du.mean()), effect_size=float((dw.mean() - du.mean()) / pooled),
                auc=float(roc_auc_score(lab, np.r_[dw, du])), p=float(t.pvalue))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--unlearn_epochs", type=int, default=5)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(4)

    c = torch.load(CACHE / "embeddings.pt")
    emb, gt, box = c["emb"], c["gt"], c["box"]
    perm = np.random.default_rng(2000 + args.seed).permutation(len(emb))
    W, R, U = np.sort(perm[:40]), np.sort(perm[40:160]), np.sort(perm[160:])
    base = SamModel.from_pretrained(c["model_id"])
    t0 = time.time()

    P = np.r_[W, R]
    print("full", flush=True)
    full = finetune(base, emb[P], gt[P], box[P], args.epochs, 1e-4, args.seed)
    print("oracle", flush=True)
    oracle = finetune(base, emb[R], gt[R], box[R], args.epochs, 1e-4, args.seed)
    print("retain-ft", flush=True)
    retain_ft = finetune(full, emb[R], gt[R], box[R], args.unlearn_epochs, 5e-5, args.seed + 1)
    print("neggrad+", flush=True)
    neg = neggrad_plus(full, emb, gt, box, W, R, args.unlearn_epochs, 5e-5, args.seed + 2)
    print(f"training took {time.time() - t0:.0f} s", flush=True)

    models = {"full": full, "retain-ft": retain_ft, "neggrad+": neg, "oracle": oracle}
    sc = {m: audit(models[m], emb, gt, box, args.k, seed=777) for m in models}

    rep = dict(split=dict(withdrawn=len(W), retain=len(R), unseen=len(U)), k=args.k, epochs=args.epochs,
               unlearn_epochs=args.unlearn_epochs, seed=args.seed, fine_tune="mask decoder only (CPU)")
    rep["dice_tight"] = {m: {"withdrawn": float(sc[m]["tight_dice"][W].mean()),
                             "retain": float(sc[m]["tight_dice"][R].mean()),
                             "unseen": float(sc[m]["tight_dice"][U].mean())} for m in models}
    rep["trace_vs_oracle"] = {m: {key: trace(sc[m], sc["oracle"], key, W, U)
                                  for key in ["tight_margin", "box_margin", "tight_dice", "box_dice", "box_stability"]}
                              for m in ["full", "retain-ft", "neggrad+"]}
    (OUT / "results_unlearning.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
