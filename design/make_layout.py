"""Generate the run manifest and test-time grids for design v2 (Medical SAM3), seeded and reproducible.

Outputs (next to this script):
  runs.csv              every model to produce (seed x modality x condition), with parents and a
                        randomized execution order
  prompt_conditions.csv test-time prompt grid, applied to every model x test image
  recovery_arms.csv     continued fine-tuning checkpoints and quantization settings
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

MASTER_SEED = 20261114               # record this; change it only before any confirmatory run
DEV_SEEDS = [0]                      # hyperparameter tuning, compute timing and variance estimates only
CONFIRM_SEEDS = [1, 2, 3, 4, 5, 6]   # confirmatory replicates; set by power_sim_v2.py
MODALITIES = {                       # dataset choices are candidates until the leakage checks pass (DESIGN Sec. 3)
    "ultrasound": "held-out (DDTI or BUS-BRA)",
    "dermoscopy": "held-out (HAM10000 or IMA++, after ISIC 2018 dedup)",
    "endoscopy": "seen-modality control (Kvasir-SEG)",
}
FLAG_FRACTIONS = [0.10, 0.50, 1.00]  # nested within each draw
FLAG_DRAWS = 3                       # SSD/SAP draws at 10% and 50%; 100% has one (draws identical)
UNLEARN_METHODS = ["SSD", "SAP", "RmF"]  # RmF = fine-tune from checkpoint minus flagged samples
BASE = "Medical SAM3 released 2D checkpoint"
OUT = Path(__file__).resolve().parent

rows = []
for seed, mod in itertools.product(DEV_SEEDS + CONFIRM_SEEDS, MODALITIES):
    split = "dev" if seed in DEV_SEEDS else "confirmatory"
    base = dict(seed=seed, modality=mod, block=f"s{seed}_{mod}", split=split, train_prompt="text")
    pre = f"s{seed}_{mod}"
    for cond, data in [("clean", "train pool, true masks"),
                       ("corrupted", "train pool, 40% with ring corruption"),
                       ("oracle", "train pool minus ALL corrupted")]:
        rows.append({**base, "model_id": f"{pre}_{cond}", "condition": cond, "method": None,
                     "flag_frac": None, "flag_draw": None, "parent": BASE, "is_finetune": True,
                     "train_data": data})
    for m, f in itertools.product(UNLEARN_METHODS, FLAG_FRACTIONS):
        if m == "RmF" and f == 1.0:
            continue  # identical to the oracle by construction; the oracle doubles as the check
        n_draws = 1 if (m == "RmF" or f == 1.0) else FLAG_DRAWS
        for draw in range(1, n_draws + 1):
            ft = m == "RmF"
            rows.append({**base, "model_id": f"{pre}_{m}_f{int(f * 100):03d}_d{draw}",
                         "condition": "unlearned", "method": m, "flag_frac": f, "flag_draw": draw,
                         "parent": BASE if ft else f"{pre}_corrupted", "is_finetune": ft,
                         "train_data": ("train pool minus flagged subset" if ft
                                        else "corrupted model + flagged subset (+ trusted set for SAP)")})
runs = pd.DataFrame(rows)

# Randomize execution order within dependency tiers (fine-tunes first, then post-hoc unlearning),
# so no condition or modality lines up with calendar time (code changes, driver updates).
rng = np.random.default_rng(MASTER_SEED)
runs["tier"] = np.where(runs.is_finetune, 1, 2)
runs["exec_order"] = 0
for tier in (1, 2):
    idx = runs.index[runs.tier == tier]
    runs.loc[idx, "exec_order"] = rng.permutation(len(idx)) + 1
runs = runs.sort_values(["tier", "exec_order"]).reset_index(drop=True)
runs.to_csv(OUT / "runs.csv", index=False)

# Test-time prompt grid. Training prompt is text, so text_canonical is the reference level for H4.
# Synonyms are chosen per modality and frozen before seeds 1-6 (DESIGN Sec. 9); not invented here.
P = []
for t in ["canonical", "synonym_1", "synonym_2"]:
    P.append(dict(prompt_type="text", text_variant=t, box_expand=None, n_points=None, point_shift=None, draws=1))
for e in [0.0, 0.10, 0.20, 0.30]:          # box expansion as a fraction of the box side; 0 = tight
    P.append(dict(prompt_type="box", text_variant=None, box_expand=e, n_points=None, point_shift=None, draws=1))
for n, s in itertools.product([1, 3, 5], [0.0, 0.10, 0.25]):  # shift as a fraction of the lesion radius
    P.append(dict(prompt_type="point", text_variant=None, box_expand=None, n_points=n, point_shift=s,
                  draws=1 if (n == 1 and s == 0.0) else 3))
P.append(dict(prompt_type="text+box", text_variant="canonical", box_expand=0.0, n_points=None, point_shift=None, draws=1))
prompts = pd.DataFrame(P)
prompts["geometric"] = prompts.prompt_type.isin(["box", "point"])   # H4 primary contrast: geometric vs text
prompts.to_csv(OUT / "prompt_conditions.csv", index_label="prompt_id")

# Recovery arms (applied to every unlearned model and to the oracle as the drift reference).
arms = [dict(arm="finetune", setting=f"step_{s}", detail="continued fine-tuning on trusted set, text prompts")
        for s in [0, 100, 300, 1000]]
arms += [dict(arm="quantization", setting=q, detail="post-training weight quantization (secondary arm)")
         for q in ["int8", "int4"]]
pd.DataFrame(arms).to_csv(OUT / "recovery_arms.csv", index=False)

conf = runs[runs.split == "confirmatory"]
print(f"models total={len(runs)} confirmatory={len(conf)} fine-tunes={int(runs.is_finetune.sum())} "
      f"(confirmatory {int(conf.is_finetune.sum())})")
print("confirmatory unlearned per block:", int((conf.condition == "unlearned").sum() / conf.block.nunique()),
      "| blocks:", conf.block.nunique())
print("prompt conditions:", len(prompts), "| prompt evaluations per image per model:", int(prompts.draws.sum()))
