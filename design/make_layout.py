"""Generate the run manifest for the unlearning-audit study (seeded, reproducible).

Outputs (next to this script):
  runs.csv              every model to produce, with parent dependencies and a randomized execution order
  prompt_conditions.csv test-time prompt-perturbation grid (applied to every model x test image)
  recovery_finetune.csv continued-fine-tuning checkpoints (applied to every unlearned/oracle model)
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

MASTER_SEED = 20261114          # record this; change it only before any confirmatory run
DEV_SEEDS = [0]                 # used for hyperparameter tuning and pipeline debugging only
CONFIRM_SEEDS = [1, 2, 3, 4, 5, 6]  # confirmatory replicates (the unit of replication); set by power_sim*.py
FLAG_DRAWS = 5                  # independent flagged-subset draws per SSD/SAP condition (post-hoc, no retraining)
FLAG_FRACTIONS = [0.10, 0.25, 0.50, 1.00]   # nested: 10% subset of 25% subset of ...
UNLEARN_METHODS = ["SSD", "SAP", "RmF"]     # RmF = retrain from scratch minus flagged samples
OUT = Path(__file__).resolve().parent

rows = []
for seed in DEV_SEEDS + CONFIRM_SEEDS:
    split = "dev" if seed in DEV_SEEDS else "confirmatory"
    base = dict(seed=seed, split=split)
    rows.append({**base, "model_id": f"s{seed}_clean", "condition": "clean",
                 "method": None, "flag_frac": None, "parent": None, "trains_from_scratch": True,
                 "train_data": "all train images, original masks"})
    rows.append({**base, "model_id": f"s{seed}_corrupt", "condition": "corrupted",
                 "method": None, "flag_frac": None, "parent": None, "trains_from_scratch": True,
                 "train_data": "all train images, 40% with ring"})
    rows.append({**base, "model_id": f"s{seed}_oracle", "condition": "oracle",
                 "method": None, "flag_frac": None, "parent": None, "trains_from_scratch": True,
                 "train_data": "train images minus ALL corrupted"})
    for m, f in itertools.product(UNLEARN_METHODS, FLAG_FRACTIONS):
      scratch = m == "RmF"
      for draw in range(1, 2 if (scratch or f == 1.0) else FLAG_DRAWS + 1):  # 100% flagged: draws would be identical
        rows.append({**base, "model_id": f"s{seed}_{m}_f{int(f*100):03d}_d{draw}", "condition": "unlearned",
                     "method": m, "flag_frac": f, "flag_draw": draw,
                     "parent": None if scratch else f"s{seed}_corrupt",
                     "trains_from_scratch": scratch,
                     "train_data": ("train images minus flagged subset" if scratch
                                    else "corrupted model + flagged subset + trusted set")})
runs = pd.DataFrame(rows)

# Randomize execution order within dependency tiers so no condition is aligned with
# calendar time (code changes, driver updates, GPU nondeterminism drift).
rng = np.random.default_rng(MASTER_SEED)
runs["tier"] = np.where(runs["parent"].isna(), 1, 2)
runs["exec_order"] = 0
for tier in (1, 2):
    idx = runs.index[runs["tier"] == tier]
    runs.loc[idx, "exec_order"] = rng.permutation(len(idx)) + 1
runs = runs.sort_values(["tier", "exec_order"]).reset_index(drop=True)
runs.to_csv(OUT / "runs.csv", index=False)

# Test-time prompt perturbation: within-model, fully crossed, deterministic per image.
prompts = []
for jitter in [0.0, 0.10, 0.20, 0.30]:       # box expansion as fraction of box side
    prompts.append(dict(prompt_type="box", n_points=None, box_expand=jitter, point_shift=None))
for n_pts, shift in itertools.product([1, 3, 5], [0.0, 0.10, 0.25]):  # shift as fraction of polyp radius
    prompts.append(dict(prompt_type="points", n_points=n_pts, box_expand=None, point_shift=shift))
prompts.append(dict(prompt_type="box+points", n_points=1, box_expand=0.20, point_shift=0.0))
pd.DataFrame(prompts).assign(draws_per_image=5).to_csv(OUT / "prompt_conditions.csv", index_label="prompt_id")

# Continued fine-tuning on the trusted set: checkpoints along one trajectory per model.
pd.DataFrame({"step": [0, 100, 300, 1000, 3000], "data": "trusted set (clean)",
              "lr": "same as final training stage"}).to_csv(OUT / "recovery_finetune.csv", index=False)

conf = runs[runs.split == "confirmatory"]
print(f"models total={len(runs)} confirmatory={len(conf)} from-scratch trainings={int(runs.trains_from_scratch.sum())}")
print(conf.groupby("condition").size().to_string())
print("prompt conditions:", len(prompts))
