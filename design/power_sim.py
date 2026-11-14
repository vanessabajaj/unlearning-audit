"""Simulation-based power for the unlearning-audit design (no pilot data: assumptions are explicit).

H3 (primary): among unlearned models that "look clean" at the output, does the audit score A
predict re-emergence R beyond output-level metrics M0?  Test = coefficient on A in a model-level
regression  R ~ A + M0 + seed  (seed as fixed block), one per recovery arm.
Two recovery arms are co-primary  ->  alpha = 0.05 / 2 = 0.025 each (Bonferroni).

Stringent variant adds condition (method x flag fraction) fixed effects, so A must predict
re-emergence *within* a method and flag fraction, not just by tracking which method was used.

H1 (positive control): corrupted vs oracle audit score, paired by seed -> paired t-test, closed form.

Data-generating process (standardized units), per seed s, condition c (12), one model each:
    m      = condition effect + seed effect + model-specific noise   (latent residual memory)
             variance shares: condition q_c, seed q_s, within 1 - q_c - q_s
    M0     = rho0 * m + sqrt(1 - rho0^2) * e0     (output-level metric; weak by premise)
    A      = rhoA * m + sqrt(1 - rhoA^2) * eA     (audit score)
    R      = rhoR * m + sqrt(1 - rhoR^2) * eR     (model-level mean re-emergence, per arm)
"Looks clean" filter: keep models whose M0 is below the p_clean quantile within the seed.
Null check: A carries no information about m (rhoA = 0) but correlates with M0.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
N_COND = 12
ALPHA = 0.025
N_SIMS = 2000


def simulate_once(rng, n_seeds, rhoA, p_clean, stringent, rho0=0.3, rhoR=0.8,
                  q_c=0.4, q_s=0.2, null=False):
    cond_eff = rng.normal(0, np.sqrt(q_c), N_COND)
    seed_eff = rng.normal(0, np.sqrt(q_s), n_seeds)
    m = (cond_eff[None, :] + seed_eff[:, None]
         + rng.normal(0, np.sqrt(1 - q_c - q_s), (n_seeds, N_COND)))
    M0 = rho0 * m + np.sqrt(1 - rho0**2) * rng.standard_normal(m.shape)
    if null:   # A tracks the output metric but has no extra information about m
        A = 0.5 * M0 + np.sqrt(1 - 0.25) * rng.standard_normal(m.shape)
    else:
        A = rhoA * m + np.sqrt(1 - rhoA**2) * rng.standard_normal(m.shape)
    R = rhoR * m + np.sqrt(1 - rhoR**2) * rng.standard_normal(m.shape)

    k = max(2, int(round(p_clean * N_COND)))           # models kept per seed
    keep = np.argsort(M0, axis=1)[:, :k]                # lowest output-level residual = looks clean
    rows = np.repeat(np.arange(n_seeds), k)
    cols = keep.ravel()
    y = R[rows, cols]
    X = [A[rows, cols], M0[rows, cols]]
    X += [(rows == s).astype(float) for s in range(n_seeds)]        # seed blocks (intercepts)
    if stringent:
        X += [(cols == c).astype(float) for c in range(1, N_COND)]  # condition FE
    X = np.column_stack(X)
    X = X[:, X.any(axis=0)]   # drop conditions never kept
    # Remaining dummy collinearity is harmless: the A coefficient stays estimable, and
    # lstsq + pinv give its correct estimate and variance; df uses the true rank.
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = len(y) - np.linalg.matrix_rank(X)
    if df < 3:
        return False
    resid = y - X @ beta
    s2 = resid @ resid / df
    cov = s2 * np.linalg.pinv(X.T @ X)
    t = beta[0] / np.sqrt(cov[0, 0])
    return 2 * stats.t.sf(abs(t), df) < ALPHA


def power(n_seeds, rhoA, p_clean, stringent, null=False, seed=0, n_sims=N_SIMS):
    rng = np.random.default_rng(seed)
    hits = sum(simulate_once(rng, n_seeds, rhoA, p_clean, stringent, null=null) for _ in range(n_sims))
    p = hits / n_sims
    half = 1.96 * np.sqrt(p * (1 - p) / n_sims)
    return p, max(0, p - half), min(1, p + half)


if __name__ == "__main__":
    seeds_grid = [3, 4, 5, 6, 8, 10, 12, 15]
    rows = []
    for i, (S, rhoA, pc, strict) in enumerate(itertools.product(seeds_grid, [0.4, 0.5, 0.6, 0.7],
                                                               [0.5, 0.75], [False, True])):
        p, lo, hi = power(S, rhoA, pc, strict, seed=1000 + i)
        rows.append(dict(n_seeds=S, rhoA=rhoA, p_clean=pc,
                         analysis="stringent" if strict else "primary", power=p, mc_lo=lo, mc_hi=hi))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "power_results.csv", index=False)

    # Type I error check under the null (A uninformative beyond M0)
    t1 = []
    for S, strict in itertools.product([3, 5, 10], [False, True]):
        p, lo, hi = power(S, 0.0, 0.75, strict, null=True, seed=9000 + S + 100 * strict, n_sims=4000)
        t1.append(dict(n_seeds=S, analysis="stringent" if strict else "primary", type1=p, mc_lo=lo, mc_hi=hi))
    pd.DataFrame(t1).to_csv(OUT / "power_type1.csv", index=False)

    # H1 closed form: paired t on seed-level differences, alpha 0.05 two-sided
    from statsmodels.stats.power import TTestPower
    h1 = {S: round(TTestPower().solve_power(nobs=S, alpha=0.05, power=0.8), 2) for S in [3, 4, 5, 6, 8]}
    (OUT / "power_h1_mde.json").write_text(json.dumps({"mde_dz_at_80pct_power_by_seeds": h1}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, analysis in zip(axes, ["primary", "stringent"]):
        sub = res[(res.analysis == analysis) & (res.p_clean == 0.75)]
        for rhoA, g in sub.groupby("rhoA"):
            ax.plot(g.n_seeds, g.power, marker="o", label=f"audit reliability ρA = {rhoA}")
            ax.fill_between(g.n_seeds, g.mc_lo, g.mc_hi, alpha=0.15)
        ax.axhline(0.8, ls="--", c="gray", lw=1)
        ax.set_title(f"H3 {analysis} (75% of models look clean)")
        ax.set_xlabel("confirmatory seeds")
    axes[0].set_ylabel(f"power (α = {ALPHA}, two-sided)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "power_curve.png", dpi=150)

    print(res.pivot_table(index=["analysis", "p_clean", "rhoA"], columns="n_seeds", values="power").round(2).to_string())
    print(pd.DataFrame(t1).round(3).to_string(index=False))
    print("H1 MDE d_z:", h1)
