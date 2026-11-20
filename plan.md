# Plan: Box-prompt audits of patient withdrawal in MedSAM

**Status:** draft, 2026-11-20. Not yet approved. Before Phase 1 starts, decide whether this replaces
design v2 (`design/DESIGN.md`) or becomes one arm of it (see Section 1).

## 0. Question

A hospital fine-tunes MedSAM on its own colonoscopy frames. A patient then withdraws consent and
their frames are unlearned.

- **Q1.** Can several plausible box prompts on the same frames reveal a training trace that the
  standard tight box misses?
- **Q2.** For the same number of queries per frame, do varied boxes reveal more than ordinary image
  augmentations?

**Why it is open.** A quick search (2026-11-20, three queries, not a systematic review) found:

- no membership-inference work on SAM or MedSAM;
- medical-segmentation unlearning papers that evaluate only output scores (Erase to Retain,
  arXiv 2511.16574; To forget is to preserve, arXiv 2606.16180);
- augmentation-based unlearning audits only for classifiers (A-LiRA).

Using the prompt as an audit channel is specific to promptable models. Before writing it up, run a
proper literature check with the `literature-review` skill.

## 1. How it relates to the current project

| | Design v2 | This plan |
|---|---|---|
| Model | Medical SAM3, text prompts | MedSAM (SAM ViT-B), box prompts |
| Reason for unlearning | Corrective (corrupted masks) | Privacy (patient withdrawal) |
| Outcome | Re-emergence of the error | Residual membership signal |
| Card A (cross-modality, Medical SAM3) | Covered | Not covered |

**Option A: arm of v2.** Run Phases 1–4 on the v2 endoscopy control, with Medical SAM3's box prompts
in place of MedSAM. This keeps card A.

**Option B: pivot.** Run this plan as written. It is smaller and sharper, but it drops card A.

Ask the supervisor which option applies. The rest of this plan is written for Option B. Only the
model in Section 3 changes under Option A.

## 2. Hypotheses

The audit statistic is computed per patient (Section 5). All tests are two-sided.

| ID | Claim | Role |
|---|---|---|
| H0 | On the fine-tuned model before unlearning, the K-box audit separates training patients from held-out patients better than chance | Sanity check. If H0 fails, stop and report the null. |
| H1 | Before unlearning, at the same K, K-box beats K-augmentation (AUC and TPR at 5% FPR) | Confirmatory |
| H2 | After unlearning, the K-box audit tells the unlearned model apart from the retrained oracle on the withdrawn patients; the single tight box does not | **Primary** |
| H3 | After unlearning, at the same K, K-box beats K-augmentation in the H2 comparison | Confirmatory |
| E1–E3 | Box + augmentation combined; K curve (1, 4, 8, 16); size of the box perturbation | Exploratory |

Multiplicity: H1–H3 are tested by Holm at family-wise α = 0.05. H0 is a gate, not part of the
family.

## 3. Models

- **Base:** the released MedSAM ViT-B checkpoint, with version and hash recorded. Check that MedSAM's
  own training data includes the chosen colonoscopy sets. If it does, the "held-out" patients were
  seen during pretraining, so use a dataset that is absent from MedSAM's training list.
- **Per seed, models trained from the base:**
  - `full`: fine-tuned on the train pool including the withdrawn set W;
  - `oracle`: fine-tuned on the train pool minus W, with the same recipe and seed. This is the
    reference that the unlearned models must match.
- **Unlearning methods, applied to `full`:**
  - `retain-ft`: continued fine-tuning on the retain set only (cheap baseline);
  - `NegGrad+`: gradient ascent on W plus descent on the retain set;
  - `SSD`: selective synaptic dampening, carried over from v2.
- **Positive control:** `full` itself (no unlearning). The audit must flag this model.
- **Fine-tune recipe:** set it on the dev seed with box prompts, recording MedSAM's training
  box-jitter range. The audit's box family must go beyond that range (Section 4).

## 4. Query families

Each family is applied to every frame with K ∈ {1, 4, 8, 16}. K = 1 in every family is the tight
ground-truth box, so all families share the same baseline.

| Family | Queries |
|---|---|
| `box` | Sample (shift, scale, aspect) per query: centre shift up to 0.3 × box side, scale 0.8–1.5, aspect 0.8–1.25. Clip to the image. The "plausible" bound is that the box still covers at least 70% of the lesion. Seeded and identical across models. |
| `aug` | Tight box, transformed together with the image: horizontal or vertical flip, rotation of ±15°, crop-resize at 0.85–1.0, colour jitter. Seeded and identical across models. |
| `box+aug` | One draw of each per query (exploratory, E1) |

**Two budgets.** The primary budget is the number of queries. Compute is reported as a secondary
budget. A box query reuses the image embedding, so only the prompt encoder and decoder rerun. An
augmentation query re-runs the ViT-B encoder. Box queries are therefore far cheaper per query.
Measure the ratio on the dev seed and report it with H1 and H3.

## 5. Audit statistics

**Per frame**, for the K masks:

- mean Dice against the ground truth;
- mean pre-threshold logit margin inside versus outside the lesion;
- stability: the mean pairwise IoU of the K masks.

**Per patient:** the mean over frames. The primary statistic, fixed before seed 1, is the logit
margin. The other two are secondary.

**Calibration** uses the reference-model approach, a light version of LiRA:

- train R reference models without the target patients (the oracle-type models across seeds);
- convert each patient's score to a z-score against the reference distribution.

**Primary comparison (H2, H3).** For the withdrawn patients, compare the unlearned model's z-scores
with the oracle's z-scores. Report:

- the effect size (paired difference in z, with a CI over seeds);
- AUC for telling unlearned from oracle;
- TPR at 5% FPR. With about 30 patients, 1% FPR cannot be estimated.

## 6. Data and units

- **Unit = patient**, approximated by the colonoscopy sequence. The candidate dataset is
  CVC-ClinicDB, which provides sequence IDs; verify the count and that it maps to patients.
- PolypGen is a candidate for a second or larger set. Check whether it has patient or sequence IDs.
- **Kvasir-SEG has no patient IDs.** Use it only as a secondary set, with the near-duplicate groups
  from `src/dedup.py` standing in for patients. Fix the hash-only linking rule first (open item from
  the grouping run).
- **Splits** are made at patient or group level with `src/make_splits.py`:
  - train pool, containing the withdrawable patients;
  - held-out patients, the non-members for H0 and H1;
  - test patients, for utility (Dice on the retain set and test set).
- **Withdrawn set W:** drawn from the train pool. Use 1 patient and 3 patients, with 3 draws per
  seed.
- **Leakage check.** Every near-duplicate of a withdrawn frame must also leave the retain set. If it
  stays, the audit measures copy leakage, not memory.

## 7. Replication and power

- **Replicate = seed** (fine-tune seed, data order, W draw). Frames and patients within a seed are
  repeated measures.
- Number of models per seed: 2 + 3 unlearned = 5 per W draw. The reference models are reused.
- The seed count is set by a power simulation (`design/power_sim_withdrawal.py`, to be written) on
  dev-seed variance estimates. The planning default is 6 confirmatory seeds plus 1 dev seed, to be
  revised after Phase 2.
- **The smallest effect of interest is fixed before seed 1.** The default is a paired z-difference
  of 0.5 between unlearned and oracle, averaged over the withdrawn patients.

## 8. Phases

| Phase | Work | Output | Gate |
|---|---|---|---|
| 0 | Supervisor decision (Option A or B); literature check | Decision note, `refs/references.bib` updated | Go or no-go on the angle |
| 1 | Data: obtain CVC-ClinicDB, patient-level splits, dedup leakage check | `data/README.md`, splits CSV, test passing | No patient spans splits |
| 2 | Dev seed: fine-tune recipe, reproduce MedSAM baseline Dice, compute timing, variance estimates | Dev log, timing table | Fine-tuned Dice within the expected range |
| 3 | Power simulation on the dev estimates; freeze seeds, K values, statistics, box ranges | `power_sim_withdrawal.py`, frozen config | Preregistration committed |
| 4 | Confirmatory runs: models, unlearning, all query families | Score tables per seed | H0 gate |
| 5 | Analysis H1–H3, exploratory E1–E3, figures | Results notebook, figures | – |
| 6 | Write-up | Report or paper draft | – |

## 9. Reuse from the repository

- `src/dedup.py` and `src/make_splits.py`: patient or group-level splits and leakage checks.
- `design/make_layout.py`: pattern for the seeded run manifest. The box grid in
  `prompt_conditions.csv` is the starting point for the `box` family.
- The seed-as-replicate design and the simulation-based power approach from v2.
- The Kvasir-SEG download and checksum in `data/README.md`.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Fine-tuned foundation models leak little membership signal, so H0 fails | H0 is a gate. Report the null with its MDE. The ranking of box against augmentation can still be reported descriptively. |
| MedSAM's box jitter makes the model invariant to plausible boxes | Box ranges go beyond the recorded training jitter. E3 sweeps the magnitude. |
| Too few patients for low-FPR metrics | Report TPR at 5% FPR and z-score effect sizes, not 0.1% FPR. Add PolypGen if it has IDs. |
| Near-duplicates across patients | Dedup groups are merged with patient IDs before splitting. |
| Evaluation datasets were in MedSAM pretraining | Check the MedSAM data list. Prefer a dataset outside it. |
| Comparing only against held-out patients overstates the residue | The primary comparison is against the oracle, not against non-members. |

## 11. Open questions

1. Option A or Option B (supervisor).
2. Can the patient-level unit be justified with sequence IDs, given that some patients may span
   several sequences?
3. GPU budget: measure on the dev seed. The plan assumes ViT-B fine-tunes of under one hour each.
