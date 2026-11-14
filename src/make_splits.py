"""Assign near-duplicate groups (from dedup.py) to data partitions.

Whole groups go to one partition, so no polyp appears on both sides of a split.
Groups are placed greedily (largest first, ties in seeded random order) into the partition
with the largest remaining deficit, weighted by polyp-size tercile, so every partition gets
roughly its target share of small, medium and large polyps.

Usage:
  python src/make_splits.py --groups data/groups/groups.csv --out data/splits --seed 20261114
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = {"train_pool": 0.65, "trusted": 0.10, "val": 0.10, "test": 0.15}


def add_terciles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["size_tercile"] = pd.qcut(df.mask_area_frac.rank(method="first"), 3, labels=["small", "medium", "large"])
    return df


def assign(df: pd.DataFrame, targets=TARGETS, seed: int = 0) -> pd.DataFrame:
    if not np.isclose(sum(targets.values()), 1):
        raise ValueError("targets must sum to 1")
    df = add_terciles(df)
    terc = ["small", "medium", "large"]
    comp = (pd.crosstab(df.group_id, df.size_tercile).reindex(columns=terc, fill_value=0))
    rng = np.random.default_rng(seed)
    order = comp.assign(n=comp.sum(1), r=rng.random(len(comp))).sort_values(["n", "r"], ascending=[False, True]).index
    tc = df.size_tercile.value_counts().reindex(terc).to_numpy()
    want = {s: f * tc for s, f in targets.items()}            # target count per partition x tercile
    have = {s: np.zeros(3) for s in targets}
    where = {}
    for g in order:
        v = comp.loc[g].to_numpy()
        # deficit in the tercile(s) this group would fill, normalized by partition size
        score = {s: ((want[s] - have[s]) * v).sum() / (want[s].sum() + 1e-9) for s in targets}
        best = max(score, key=score.get)
        where[g] = best
        have[best] += v
    df["split"] = df.group_id.map(where)
    return df


def check(df: pd.DataFrame, targets=TARGETS) -> dict:
    spans = df.groupby("group_id").split.nunique()
    if (spans > 1).any():
        raise AssertionError(f"{int((spans > 1).sum())} groups span multiple splits")
    n = len(df)
    summary = {s: {"n_images": int((df.split == s).sum()),
                   "frac": round(float((df.split == s).mean()), 4),
                   "target": targets[s],
                   "n_groups": int(df.loc[df.split == s, "group_id"].nunique()),
                   "terciles": df.loc[df.split == s, "size_tercile"].value_counts().reindex(
                       ["small", "medium", "large"]).astype(int).to_dict()}
               for s in targets}
    summary["_total_images"] = n
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = assign(pd.read_csv(a.groups), seed=a.seed)
    summary = check(df)
    summary["_seed"] = a.seed
    df.drop(columns=["image_path", "mask_path"]).to_csv(a.out / "splits.csv", index=False)
    (a.out / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
