"""Simulation-based power for design v2 (Medical SAM3, 3 modalities). No pilot data: every input
is an assumption, swept for sensitivity. Re-run with seed-0 estimates before freezing the seed count.

Structure (matches make_layout.py): a block = one seed x modality, i.e. an independent set of
fine-tunes on one dataset. Per block: 8 unlearning conditions (RmF 10/50%; SSD and SAP at 10/50%
with `draws` flagged-subset draws each, and at 100% with one) = 2 + 2*(2*draws + 1) models.
Three co-primary tests (H3 prompt arm, H3 fine-tune arm, H4) -> alpha = 0.05 / 3 each.

H3: among models that look clean (lowest 75% of output metric M0 within a block), the coefficient
    on audit score A in  R ~ A + M0 + block  (block fixed effects).
    New vs v1: draws of one condition can share variance (they come from the same corrupted
    parent), controlled by `draw_share` = fraction of the model-specific variance shared by draws.
H4: per model, D = mean re-emergence under geometric prompts minus under text (over test images).
    D = delta_modality + block effect + model noise. Test: one-sample t on block means of D.
    Scenarios: effect equal in all modalities, or present in only one modality.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
ALPHA = 0.05 / 3
N_MOD = 3


def block_conditions(draws):
    # condition index per model: 0,1 = RmF 10/50; 2,3 = SSD/SAP 100 (one model); 4-7 = SSD/SAP 10/50 (draws)
    return np.array([c for c in range(8) for _ in range(1 if c < 4 else draws)])


def h3_once(rng, n_seeds, rhoA, draws, draw_share, p_clean=0.75, rho0=0.3, rhoR=0.8,
            q_c=0.4, q_b=0.2, null=False):
    cond = block_conditions(draws)
    n_blocks, n = n_seeds * N_MOD, len(cond)
    # condition effects differ somewhat by modality (main effect + interaction)
    ce = rng.normal(0, np.sqrt(q_c * 0.7), 8)[None, :] + rng.normal(0, np.sqrt(q_c * 0.3), (N_MOD, 8))
    mod_of_block = np.repeat(np.arange(N_MOD), n_seeds)
    q_w = 1 - q_c - q_b
    shared = rng.normal(0, np.sqrt(q_w * draw_share), (n_blocks, 8))[:, cond]      # parent-shared part
    own = rng.normal(0, np.sqrt(q_w * (1 - draw_share)), (n_blocks, n))
    m = ce[mod_of_block][:, cond] + rng.normal(0, np.sqrt(q_b), (n_blocks, 1)) + shared + own
    M0 = rho0 * m + np.sqrt(1 - rho0**2) * rng.standard_normal(m.shape)
    if null:
        A = 0.5 * M0 + np.sqrt(0.75) * rng.standard_normal(m.shape)
    else:
        A = rhoA * m + np.sqrt(1 - rhoA**2) * rng.standard_normal(m.shape)
    R = rhoR * m + np.sqrt(1 - rhoR**2) * rng.standard_normal(m.shape)
    k = int(round(p_clean * n))
    keep = np.argsort(M0, axis=1)[:, :k]
    rows = np.repeat(np.arange(n_blocks), k)
    cols = keep.ravel()
    X = np.column_stack([A[rows, cols], M0[rows, cols]] + [(rows == b).astype(float) for b in range(n_blocks)])
    y = R[rows, cols]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = len(y) - np.linalg.matrix_rank(X)
    r = y - X @ beta
    se = np.sqrt(r @ r / df * np.linalg.pinv(X.T @ X)[0, 0])
    return 2 * stats.t.sf(abs(beta[0] / se), df) < ALPHA


def h4_once(rng, n_seeds, delta, scenario, draws=3, block_share=0.3):
    n_models = len(block_conditions(draws))
    n_blocks = n_seeds * N_MOD
    mod_of_block = np.repeat(np.arange(N_MOD), n_seeds)
    if scenario == "all":
        d_mod = np.full(N_MOD, delta)
    elif scenario == "one_modality":          # same average effect, concentrated in one modality
        d_mod = np.array([delta * N_MOD, 0.0, 0.0])
    else:                                       # null
        d_mod = np.zeros(N_MOD)
    D = (d_mod[mod_of_block][:, None]
         + rng.normal(0, np.sqrt(block_share), (n_blocks, 1))
         + rng.normal(0, np.sqrt(1 - block_share), (n_blocks, n_models)))
    bm = D.mean(axis=1)
    return stats.ttest_1samp(bm, 0.0).pvalue < ALPHA


def power(fn, n_sims, seed, **kw):
    rng = np.random.default_rng(seed)
    p = np.mean([fn(rng, **kw) for _ in range(n_sims)])
    return p, 1.96 * np.sqrt(p * (1 - p) / n_sims)


if __name__ == "__main__":
    seeds = [3, 4, 5, 6, 8]
    rows = []
    for i, (S, rhoA, draws, ds) in enumerate(itertools.product(seeds, [0.4, 0.5, 0.6], [1, 3], [0.0, 0.5])):
        p, h = power(h3_once, 2000, 100 + i, n_seeds=S, rhoA=rhoA, draws=draws, draw_share=ds)
        rows.append(dict(test="H3", n_seeds=S, rhoA=rhoA, draws=draws, draw_share=ds, power=p, mc_halfwidth=h))
    for i, (S, d, sc) in enumerate(itertools.product(seeds, [0.1, 0.2, 0.3, 0.5], ["all", "one_modality"])):
        p, h = power(h4_once, 4000, 5000 + i, n_seeds=S, delta=d, scenario=sc)
        rows.append(dict(test="H4", n_seeds=S, delta_sd=d, scenario=sc, power=p, mc_halfwidth=h))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "power_results_v2.csv", index=False)

    t1 = []
    for S in [3, 6]:
        p, h = power(h3_once, 4000, 900 + S, n_seeds=S, rhoA=0.0, draws=3, draw_share=0.5, null=True)
        t1.append(dict(test="H3", n_seeds=S, type1=p, mc_halfwidth=h))
        p, h = power(h4_once, 8000, 950 + S, n_seeds=S, delta=0.0, scenario="null")
        t1.append(dict(test="H4", n_seeds=S, type1=p, mc_halfwidth=h))
    pd.DataFrame(t1).to_csv(OUT / "power_type1_v2.csv", index=False)

    h3 = res[res.test == "H3"]
    print("H3 power (alpha 0.0167); rows rhoA/draws/draw_share, cols seeds")
    print(h3.pivot_table(index=["rhoA", "draws", "draw_share"], columns="n_seeds", values="power").round(2).to_string())
    h4 = res[res.test == "H4"]
    print("\nH4 power (alpha 0.0167); delta in SD of model-level D")
    print(h4.pivot_table(index=["scenario", "delta_sd"], columns="n_seeds", values="power").round(2).to_string())
    print("\nType I:")
    print(pd.DataFrame(t1).round(4).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    sub = h3[(h3.draws == 3) & (h3.draw_share == 0.5)]
    for rhoA, g in sub.groupby("rhoA"):
        axes[0].errorbar(g.n_seeds, g.power, yerr=g.mc_halfwidth, marker="o", capsize=2, label=f"ρA = {rhoA}")
    axes[0].set_title("H3: audit adds prediction (3 draws, draw share 0.5)")
    sub = h4[h4.scenario == "all"]
    for d, g in sub.groupby("delta_sd"):
        axes[1].errorbar(g.n_seeds, g.power, yerr=g.mc_halfwidth, marker="o", capsize=2, label=f"δ = {d} SD")
    axes[1].set_title("H4: geometric vs text re-emergence (all modalities)")
    for ax in axes:
        ax.axhline(0.8, ls="--", c="gray", lw=1)
        ax.set_xlabel("confirmatory seeds (x 3 modalities)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("power (α = 0.017, two-sided)")
    fig.tight_layout()
    fig.savefig(OUT / "power_curve_v2.png", dpi=150)
