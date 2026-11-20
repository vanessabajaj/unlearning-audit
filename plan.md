# Project plan: Box-prompt audits of patient withdrawal in MedSAM

*Draft for group discussion. Version 2, 2026-11-20.*

---

## 1. Summary

A hospital fine-tunes MedSAM on its own colonoscopy frames. Later, a patient withdraws consent, so
we remove ("unlearn") their frames from the model.

We want to know whether the model still carries a trace of that patient. The usual check prompts
each frame once, with a tight box around the polyp. Polyps are often easy to segment, so every model
scores well on a tight box and any remaining trace is hidden.

Our idea is to query each frame with **many plausible boxes**: shifted, enlarged or loose. Loose
boxes force the model to rely on what it learned about that specific image. A model that still
remembers a frame should respond more sharply and more consistently than a model that never saw it.

We also test whether varied boxes reveal more than ordinary image augmentations (flips, crops,
colour changes) when both use **the same number of queries**. If they do, the prompt is an audit
channel that only promptable models have.

---

## 2. Research questions

- **RQ1.** After simulated patient withdrawal, do several plausible box prompts on the same frames
  reveal a remaining training signal that the single standard box misses?
- **RQ2.** At the same number of queries per frame, do varied boxes reveal more than image
  augmentations?
- **RQ3 (exploratory).**
  - Does the answer depend on the unlearning method?
  - Does it depend on how many queries we use?
  - Does it depend on how far the boxes are perturbed?

---

## 3. Hypotheses

The audit gives one score per patient (Section 6.5). All tests are two-sided. We test both
directions because an unlearning method can also over-forget: it can push a patient's score below
the level of a model that never saw them, and that is detectable too.

| ID | Statement | Role |
|---|---|---|
| H0 | On the fine-tuned model before any unlearning, the varied-box audit tells training patients apart from unseen patients better than chance | **Gate.** If H0 fails, the audit has no signal to work with. We stop and report a null result. |
| H1 | Before unlearning, at the same number of queries, varied boxes separate training patients from unseen patients better than augmentations | Confirmatory |
| H2 | After unlearning, varied boxes detect a difference between the unlearned model and a model retrained without the patient, where the single tight box does not | **Primary** |
| H3 | After unlearning, at the same number of queries, varied boxes detect that difference better than augmentations | Confirmatory |
| E1–E3 | Combining boxes with augmentations; the number of queries (1, 4, 8, 16); the size of the box perturbation | Exploratory |

Multiple testing: H1–H3 are corrected together with the Holm method at family-wise α = 0.05.

---

## 4. Scope decision (needs the supervisor)

This angle shares the main idea of our current design: whether unlearning holds depends on how you
prompt. It differs in model, goal and measurement.

| | Current design (v2) | This plan |
|---|---|---|
| Model | Medical SAM3, text prompts | MedSAM (SAM ViT-B), box prompts |
| Why we unlearn | Corrective: corrupted training masks | Privacy: patient withdrawal |
| What we measure | Whether the error comes back | Whether the patient's data leaves a trace |
| Assignment card A (cross-modality transfer, Medical SAM3) | Covered | Not covered |

- **Option A: add this as one arm of v2.** Run it on the endoscopy control with Medical SAM3 box
  prompts. This keeps card A. Only the model wrapper (Section 7) changes.
- **Option B: switch to this plan as the whole project.** It is smaller, sharper and cheaper, but it
  drops card A.

The rest of this document is written for Option B.

---

## 5. Definitions

| Term | Meaning here |
|---|---|
| Patient | One colonoscopy sequence, used as a stand-in for one patient |
| Withdrawn set (W) | The patients whose frames must be forgotten |
| Retain set | The training patients who stay |
| Unseen patients | Patients never used in training. They give the "not a member" reference. |
| `full` model | Fine-tuned on all training patients, including W |
| Oracle | Fine-tuned from the same starting point without W. This is what perfect unlearning should look like. |
| Query | One forward pass on one frame with one prompt, or with one augmentation |
| K | The number of queries per frame |
| Audit | Scoring frames to decide whether a model still "remembers" a patient |

---

## 6. Method

### 6.1 Data

- **Main dataset: CVC-ClinicDB** (colonoscopy frames with polyp masks and sequence labels). Before
  using sequences as patients, check:
  - how many sequences there are and how many frames each has;
  - whether the sequence labels can stand in for patients.
- **Secondary dataset: Kvasir-SEG.** It has no patient labels, so our near-duplicate groups stand in
  for patients. Before using it, remove the "hash-only" link rule in the grouping. It wrongly merged
  different polyps.
- **Optional third dataset: PolypGen**, if it provides patient or sequence labels.
- **Pretraining check.** Confirm whether MedSAM's pretraining data already contains these datasets.
  If it does, our "unseen" patients were seen during pretraining, and we should prefer a dataset that
  MedSAM did not train on.

### 6.2 Splits

All splits are made by patient. No patient may appear in two splits.

| Split | Share of patients | Used for |
|---|---|---|
| Train pool | about 55% | Fine-tuning. W is drawn from here. |
| Unseen | about 20% | Non-member reference for H0 and H1 |
| Reference pool | about 10% | Patients held out from training, used to check the calibration |
| Test | about 15% | Utility: Dice on new patients |

- **Withdrawn set W:** 1 patient (small withdrawal) and 3 patients (larger withdrawal).
- **Draws:** 3 random draws of W per seed.
- **Leakage rule:** every near-duplicate of a withdrawn frame also leaves the retain set. Otherwise
  the audit would measure copies that stayed behind, not memory.

### 6.3 Models

**Starting point:** the released MedSAM ViT-B checkpoint. Record its version and file hash.

**Fine-tuning recipe** (finalised on the development seed):

| Item | Setting |
|---|---|
| Input | 1024 × 1024 frames, box prompt taken from the ground-truth mask |
| Box jitter during training | Random, up to 20 px |
| Trained parts | Image encoder (low learning rate) and mask decoder. The prompt encoder is frozen. |
| Fallback if GPU time is short | Train the mask decoder only |
| Loss | Dice + binary cross-entropy |
| Optimiser | AdamW, cosine schedule |
| Stopping | Fixed number of epochs, chosen on the development seed |
| Fixed between `full` and oracle | Recipe, seed and data order (except that W is missing from the oracle's data) |

**Models per seed and W draw:**

| Model | How it is made |
|---|---|
| `full` | Fine-tune on the whole train pool, including W. Positive control: the audit must flag it. |
| `oracle` | Fine-tune on the train pool minus W |
| `retain-ft` | Start from `full` and continue fine-tuning on the retain set only. A cheap baseline. |
| `neggrad+` | Start from `full`. Push loss up on W while keeping loss low on the retain set. |
| `ssd` | Start from `full`. Dampen the weights that matter much more for W than for the retain set. |

**Reference models:** extra fine-tunes that leave out a known set of patients. Section 6.5 uses them
to judge what "normal" scores look like for a model that has never seen a patient.

### 6.4 Query families

Every frame is queried with each family at K = 1, 4, 8 and 16. At K = 1 every family uses the tight
box, so all three start from the same baseline. Random draws use a fixed seed, so every model sees
exactly the same queries.

**Boxes.** Each query perturbs the tight box:

| Parameter | Range |
|---|---|
| Centre shift | Up to 0.3 × box side, in a random direction |
| Scale | 0.8–1.5 |
| Aspect ratio | 0.8–1.25 |
| Clipping | To the image |
| Plausibility rule | The box must still cover at least 70% of the polyp. Draws that fail are rejected and redrawn. |
| Minimum perturbation | Beyond the ±20 px jitter used in training, so the model has not already learned to ignore it |

**Augmentations.** Each query uses the tight box and transforms the frame and the box together:

| Augmentation | Setting |
|---|---|
| Flip | Horizontal or vertical |
| Rotation | ±15° |
| Crop and resize | 85–100% of the frame |
| Colour jitter | Brightness, contrast and saturation within ±20% |

Each predicted mask is mapped back to the original frame before scoring (undo the flip, rotation and
crop).

**Combined (exploratory):** one box perturbation plus one augmentation per query.

**Cost.** A box query reuses the image embedding, so only the small prompt encoder and mask decoder
run again. An augmentation query must also run the large image encoder again. We compare at an equal
number of queries (primary) and also report GPU time per patient (secondary). That shows whether
boxes are also the cheaper audit.

### 6.5 Scoring

**Per frame**, across its K predicted masks:

| Statistic | Definition | Role |
|---|---|---|
| Logit margin | Mean pre-threshold logit inside the true polyp minus mean logit outside it, averaged over the K queries | **Primary** (fixed before the confirmatory seeds) |
| Dice | Mean Dice against the ground-truth mask | Secondary |
| Stability | Mean overlap (IoU) between every pair of the K predicted masks | Secondary |

**Per patient:** the mean over that patient's frames.

**Calibration.** For each patient, compare their score with the scores that reference models give
them without ever having seen them. Convert the score to a z-score against that reference
distribution. A high z-score means the model behaves as if it trained on that patient.

**Main contrast (H2, H3).** For each withdrawn patient, take the unlearned model's z-score minus the
oracle's z-score. Perfect unlearning gives a difference near 0.

**Measures reported:**

- the mean z-difference with a 95% confidence interval across seeds;
- AUC for telling the unlearned model from the oracle;
- true-positive rate at 5% false-positive rate. With roughly 30 patients, lower false-positive rates
  cannot be estimated reliably.

### 6.6 Statistical analysis

- **Unit of replication: the seed** (fine-tune seed, data order and W draw together). Frames and
  patients inside a seed are repeated measurements, not independent samples.
- **H0 and H1:** AUC per seed, training patients against unseen patients.
  - H0: mean AUC across seeds against 0.5.
  - H1: paired difference in AUC between the box and augmentation families, tested across seeds.
- **H2:**
  - per seed and method, the mean z-difference over the withdrawn patients;
  - a one-sample test across seeds, for the box family and for the single tight box;
  - a mixed model as a check: patient score ~ family × model + (1 | seed / patient).
- **H3:** paired difference between the box and augmentation families in the H2 statistic, tested
  across seeds.
- **Seed count:** set by a power simulation using variance measured on the development seed. The
  planning default is 1 development seed plus 6 confirmatory seeds.
- **Smallest effect we care about:** a z-difference of 0.5. Fixed before the confirmatory runs.
- **Pre-registration:** before seed 1 we freeze and commit the statistics, K values, box ranges,
  augmentations, fine-tune recipe and analysis code.

---

## 7. Implementation

### 7.1 Stack

| Item | Choice |
|---|---|
| Language and framework | Python 3.13, PyTorch |
| Model | MedSAM ViT-B checkpoint, through a thin wrapper |
| Libraries | NumPy, pandas, SciPy, statsmodels, scikit-learn (AUC), Pillow and OpenCV (image and box handling), matplotlib |
| Configs | One YAML file per experiment. Every run records its config, seed and code version (git commit). |
| Tests | pytest, run on small synthetic data so they need no GPU |

### 7.2 Code layout

Parts that already exist are marked.

```
src/
  dedup.py                 near-duplicate grouping (exists)
  make_splits.py           group-wise splits (exists; extend to patient labels)
  datasets.py              frame loader: image, mask, patient id, tight box
  medsam_wrapper.py        load checkpoint; embed(image); decode(embedding, box) -> logits
  train/finetune.py        fine-tuning loop (full, oracle, reference models)
  unlearn/
    retain_ft.py           continue fine-tuning on the retain set
    neggrad.py             loss up on W, loss down on the retain set
    ssd.py                 weight dampening by importance ratio
  audit/
    queries.py             seeded box and augmentation samplers, mask inverse mapping
    score.py               logit margin, Dice, stability per frame and per patient
    calibrate.py           z-scores against the reference models
  analysis/
    hypotheses.py          H0-H3 tests, Holm correction
    figures.py             K curves, AUC plots, cost plots
design/
  make_withdrawal_layout.py  seeded run manifest (models, W draws, execution order)
  power_sim_withdrawal.py    power simulation for the seed count
configs/                   YAML per experiment
tests/                     synthetic tests (box sampler, inverse mapping, scoring, splits)
```

### 7.3 Core audit loop (pseudo-code)

```python
for model in models_for_seed:                # full, oracle, retain-ft, neggrad+, ssd, references
    for patient in audit_patients:            # withdrawn, retain sample, unseen
        for frame in patient.frames:
            emb = model.embed(frame.image)    # computed once per frame
            for family in ["box", "aug", "box+aug"]:
                for q in queries(family, frame, K_max=16, seed=FIXED):
                    if q.needs_reencode:      # augmentation queries
                        logits = model.decode(model.embed(q.image), q.box)
                        logits = q.inverse(logits)   # back to original frame
                    else:                     # box-only queries reuse the embedding
                        logits = model.decode(emb, q.box)
                    store(model, patient, frame, family, q.index, logits)
# K = 1, 4, 8, 16 are prefixes of the same 16 queries, so one pass covers all K
scores = score.per_patient(stored, stats=["margin", "dice", "stability"])
z = calibrate.zscores(scores, reference_models)
```

Logits are stored as compact summaries (margin, Dice, a downsampled mask), not as full 1024 × 1024
maps. That keeps storage small.

### 7.4 Tests that must pass before any GPU run

- The box sampler respects the plausibility rule and returns identical boxes for the same seed.
- Each augmentation's inverse mapping recovers a synthetic mask exactly: IoU > 0.99 after
  forward-then-inverse, ignoring edge pixels under rotation.
- No patient or near-duplicate group spans two splits, and no near-duplicate of W stays in the
  retain set.
- On synthetic logits with a planted member effect, the scoring and calibration recover it. With no
  effect planted, they keep the false-positive rate near 5%.

---

## 8. Technical workflow

```
 1. Data        download -> checksum -> near-duplicate groups -> patient labels -> patient-level splits
 2. Dev seed    tune fine-tune recipe -> check baseline Dice -> time each step -> measure variance
 3. Freeze      power simulation -> choose seed count -> pre-register statistics, ranges, code
 4. Train       per seed: full + oracle + reference models  (run order randomised)
 5. Unlearn     per seed x W draw: retain-ft, neggrad+, ssd from full
 6. Utility     Dice on retain and test patients for every model (unlearning must not break the model)
 7. Audit       all families x K for every model on withdrawn, retain-sample and unseen patients
 8. Calibrate   z-scores against reference models
 9. Analyse     H0 gate -> H1 -> H2 -> H3 (Holm) -> exploratory E1-E3
10. Report      figures, tables, write-up
```

**Run order:** fine-tunes first, then unlearning, then audits. Order is randomised within each stage
so that no method or seed lines up with calendar time (code or driver changes).

**Compute:** to be measured on the development seed. The planning assumption is under 1 GPU-hour per
ViT-B fine-tune on a few hundred frames. Audits are cheap for boxes and dominated by encoder passes
for augmentations.

---

## 9. Expected results

These are **expectations before any data**, not findings. We write them down now so we can check
our reasoning afterwards.

| Comparison | What we expect | Why |
|---|---|---|
| `full` model vs unseen patients (H0) | A modest signal. Clear with varied boxes, weak with the single tight box. | A tight box saturates on easy polyps. Loose boxes expose image-specific memory. |
| Boxes vs augmentations before unlearning (H1) | Boxes ahead at the same K. The gap is larger in GPU time. | Augmentations change the image, so the model no longer sees "its" training frame exactly. Boxes keep the image intact. |
| `retain-ft` vs oracle (H2) | Small or no difference | Continued training on the retain set gradually overwrites W. |
| `neggrad+` vs oracle (H2) | A difference in the **negative** direction (over-forgetting) | Pushing the loss up on W makes the model worse on those frames than a model that never saw them. |
| `ssd` vs oracle (H2) | A positive difference: some trace left | Dampening selected weights may leave memory elsewhere in the network. |
| Boxes vs augmentations after unlearning (H3) | Boxes ahead, but the margin is less certain than in H1 | The remaining trace is smaller, so differences between families shrink. |
| Effect of K (E2) | The signal grows from K = 1 to 8, then levels off | Extra queries reduce noise until they start repeating similar boxes. |
| Box perturbation size (E3) | Strongest at moderate looseness. Weak for tiny shifts, and weak again when boxes barely cover the polyp. | Tiny shifts behave like the training jitter. Very loose boxes make every model fail. |

**If H0 fails:** the fine-tuned model leaks too little to audit. That is still a useful result. We
report the smallest effect the design could have detected, and the ranking of boxes against
augmentations as descriptive results.

---

## 10. Limitations

- **Sequences are not exactly patients.** One patient may have several sequences, or one sequence
  may mix views. Withdrawal is therefore approximate.
- **Simulated withdrawal.** We choose W at random. Real withdrawals may involve unusual patients who
  are easier or harder to forget.
- **Small numbers.** Tens of patients per dataset mean wide confidence intervals and no reliable
  low false-positive-rate estimates.
- **One modality, one model size.** Results apply to colonoscopy and MedSAM ViT-B. They may not hold
  for other modalities, larger models or text-prompted models.
- **The auditor has privileged access.** Our audit uses the withdrawn patient's frames and
  ground-truth masks, plus reference models. An outside party would have less, so we show what a
  well-informed auditor can find, not what an attacker can find.
- **No formal guarantee.** A clean audit does not prove deletion. It shows only that these queries
  find nothing.
- **Pretraining overlap.** If MedSAM was pretrained on our datasets, "unseen" patients were not
  fully unseen. This weakens the signal and blurs the contrast.
- **Approximate oracle.** The oracle differs from `full` in more than W (fewer training steps on
  other data). We match recipe and seed, but some difference remains.
- **Reference-model cost.** Calibration needs extra fine-tunes, which limits how many seeds we can
  afford.

---

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| No membership signal even before unlearning | Medium | H0 gate. Report the null with its detectable effect size. Try decoder-only fine-tuning, which may memorise more locally. |
| The model already ignores box perturbations | Medium | Perturb beyond the training jitter. Sweep the size (E3). |
| Near-duplicates leak across patients | Medium (Kvasir) | Merge near-duplicate groups with patient labels before splitting. Add a test for it. |
| GPU budget too small | Medium | Decoder-only fine-tuning. Fewer reference models. Cut the K = 16 combined family first. |
| Unlearning methods break the model | Low–medium | Utility check (step 6). Tune unlearning strength on the development seed so retain Dice stays within 2 points of `full`. |
| Augmentation inverse mapping is wrong | Low | Synthetic round-trip test (Section 7.4). |

---

## 12. Timeline (about 8 weeks)

| Week | Work | Deliverable |
|---|---|---|
| 1 | Scope decision (A or B). Data download. Patient labels and splits. | Splits file, leakage test passing |
| 2 | Model wrapper, fine-tune loop, query samplers, scoring, all synthetic tests | Code merged, tests green |
| 3 | Development seed: recipe, baseline Dice, timing, variance | Development report |
| 4 | Power simulation. Freeze and pre-register. | Frozen config committed |
| 5–6 | Confirmatory fine-tunes, unlearning, audits | Score tables for all seeds |
| 7 | Analysis and figures | Results section |
| 8 | Write-up and presentation | Report, slides |

---

## 13. Suggested task split

| Role | Owns |
|---|---|
| Data | Downloads, patient labels, near-duplicate fixes, splits, leakage tests |
| Models | MedSAM wrapper, fine-tuning, the three unlearning methods, utility checks |
| Audit | Query samplers, inverse mapping, scoring, calibration |
| Analysis | Power simulation, pre-registration, hypothesis tests, figures, write-up lead |

Everyone reviews the pre-registration before seed 1. No one looks at confirmatory results before it
is committed.

---

## 14. Deliverables

1. Code that reproduces every number from the config files and seeds.
2. Patient-level splits and the leakage report.
3. A pre-registration committed before the confirmatory runs.
4. Result tables: utility per model, audit scores per family and K, H0–H3 tests.
5. Figures:
   - signal against K for each family;
   - signal per unlearning method;
   - cost against signal;
   - signal against box-perturbation size.
6. A written report, with the expected-results table (Section 9) compared against what we actually
   found.

---

## 15. Open questions for the group

1. Option A (one arm of v2) or Option B (whole project)? This goes to the supervisor.
2. Are CVC-ClinicDB sequences a good enough stand-in for patients, or do we need PolypGen?
3. Full fine-tuning or decoder-only, given our GPU budget?
4. Do we keep the combined box + augmentation family, or cut it to save compute?
