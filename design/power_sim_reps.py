"""Follow-up: extra cheap unlearned models per seed (several flagged-subset draws per
SSD/SAP condition; no retraining) vs. more seeds. Same DGP and test as power_sim.py."""
import itertools
import numpy as np, pandas as pd
from scipy import stats
ALPHA, N_SIMS = 0.025, 2000

def once(rng, S, reps, rhoA, stringent, p_clean=0.75, rho0=0.3, rhoR=0.8, q_c=0.4, q_s=0.2, null=False):
    # 12 conditions; RmF conditions (4) get 1 model, SSD/SAP conditions (8) get `reps` draws
    # conditions 0-3 = RmF; 4,5 = SSD/SAP at 100% flagged (draws identical -> 1 model); 6-11 = SSD/SAP at 10/25/50%
    cond = np.array([c for c in range(12) for _ in range(1 if c < 6 else reps)])
    n = len(cond)
    m = rng.normal(0, np.sqrt(q_c), 12)[cond][None, :] + rng.normal(0, np.sqrt(q_s), S)[:, None] \
        + rng.normal(0, np.sqrt(1 - q_c - q_s), (S, n))
    M0 = rho0 * m + np.sqrt(1 - rho0**2) * rng.standard_normal(m.shape)
    A = (0.5 * M0 + np.sqrt(.75) * rng.standard_normal(m.shape)) if null else rhoA * m + np.sqrt(1 - rhoA**2) * rng.standard_normal(m.shape)
    R = rhoR * m + np.sqrt(1 - rhoR**2) * rng.standard_normal(m.shape)
    k = int(round(p_clean * n))
    keep = np.argsort(M0, axis=1)[:, :k]
    rows = np.repeat(np.arange(S), k); cols = keep.ravel()
    X = [A[rows, cols], M0[rows, cols]] + [(rows == s).astype(float) for s in range(S)]
    if stringent:
        X += [(cond[cols] == c).astype(float) for c in range(1, 12)]
    X = np.column_stack(X); X = X[:, X.any(0)]
    y = R[rows, cols]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = len(y) - np.linalg.matrix_rank(X)
    r = y - X @ b
    se = np.sqrt(r @ r / df * np.linalg.pinv(X.T @ X)[0, 0])
    return 2 * stats.t.sf(abs(b[0] / se), df) < ALPHA

rows = []
for i, (S, reps, rhoA, st) in enumerate(itertools.product([4, 5, 6, 8], [1, 3, 5], [0.5, 0.6], [False, True])):
    rng = np.random.default_rng(5000 + i)
    p = np.mean([once(rng, S, reps, rhoA, st) for _ in range(N_SIMS)])
    rows.append(dict(n_seeds=S, ssd_sap_draws=reps, models_per_seed=6 + 6 * reps, rhoA=rhoA,
                     analysis="stringent" if st else "primary", power=p))
res = pd.DataFrame(rows); res.to_csv("power_results_reps.csv", index=False)
print(res.pivot_table(index=["analysis", "rhoA", "ssd_sap_draws"], columns="n_seeds", values="power").round(2).to_string())
rng = np.random.default_rng(1)
for st in (False, True):
    print("type1 S=5 reps=5", "stringent" if st else "primary", np.mean([once(rng, 5, 5, 0, st, null=True) for _ in range(4000)]))
