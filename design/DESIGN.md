# Design: Prompt-conditional corrective unlearning in Medical SAM3

**Working title:** Does the fix hold? Prompt-conditional residual influence after corrective unlearning in a medical promptable segmenter.

Status: **draft v2** for pre-registration. It replaces v1 (SAM ViT-B on Kvasir-SEG only, in git history before this commit).
v1 met none of assignment card A. Why the direction changed, and the prior art behind each claim, are in
`reports/Unlearning audit novelty and gap.md`.

> **Not yet regenerated for v2:** `runs.csv`, `prompt_conditions.csv`, `recovery_finetune.csv` and the
> power results in `power_*.csv` / `power_curve.png` still describe v1. Regenerate them (Sec. 10) before freezing.

## 0. How this maps onto card A

| Card element | v2 work package |
|---|---|
| **Reproduce** | WP1: Medical SAM3's external-benchmark rows with the released checkpoint (Sec. 2) |
| **Extend: modality** | WP2: held-out ultrasound and dermoscopy datasets that Medical SAM3 did not train on, plus endoscopy as a seen-modality control (Sec. 3) |
| **Extend: prompt type** | WP2 zero-shot sweep and WP3 test-time arms: text, synonym text, point, box, text+box (Sec. 5) |
| **Contribute** | WP4: per-modality prompt-selection protocol that adds *robustness after correction*, a failure taxonomy with unlearning-specific modes, and released code (Sec. 8) |
| **Unlearning twist** | WP3: corrective unlearning of corrupted masks during fine-tuning, a pixel-level residual audit, and re-emergence by prompt type (Sec. 4-7) |

**Headline claim (narrow and testable).** In a text-trained medical promptable segmenter, corrective unlearning
that looks successful under text prompts can be bypassed by geometric (box/point) prompts. A pixel-level
internal audit taken before any perturbation predicts *where* the error re-emerges.

Proposed mechanism to test, not assume: SAM 3 separates concept presence (presence head) from localization.
Medical SAM3 was trained with text prompts only. A box prompt may route around whatever text-conditioned
unlearning changed.

**Claims we must not make.** "An audit predicts relearning" in general: J-Access (Song et al., arXiv 2608.11408)
already shows this at model level. "Unlearning is suppression, not deletion": already shown by Jang et al. and Xu et al.
Novelty rests on three things: (i) pixel localization, (ii) item/pixel-level prediction, and (iii) prompt-type and
modality dependence in a promptable segmenter.

## 1. Questions, unit, responses

**Primary question (H3).** Among unlearned models whose outputs look clean under the training prompt (text),
does a pixel-level internal audit score taken before perturbation predict re-emergence of the corrupted
error band under (a) prompt change and (b) continued fine-tuning, beyond output-level metrics?

**Co-primary question (H4).** After text-conditioned corrective unlearning, is re-emergence larger under box and
point prompts than under text prompts, and does this differ by modality?

**Unit of replication: the fine-tuning run (seed)**, within a modality. A seed fixes the corruption assignment,
the flagged subsets, data order and augmentation. Images and pixels are repeated measures. Modality is a fixed factor.

**Responses** (all per image, aggregated per model):
| Name | Definition |
|---|---|
| Dice / boundary-F | Against the true mask |
| Error-band FP (EB-FP) | Fraction of error-band pixels predicted foreground. The **error band** is the set of pixels where corrupted and true masks differ (a ring in the primary corruption) |
| Audit score A | Internal signal on the error band, standardized against the Clean reference (Sec. 6) |
| Re-emergence R | EB-FP after perturbation minus the Oracle's EB-FP under the **same** prompt/perturbation; binary if above the Oracle's 95th percentile |
| Presence score | SAM 3 presence-head output for the prompted concept (a mechanism variable for H4) |

## 2. WP1: Reproduce

- **Checkpoint:** the released Medical SAM3 2D checkpoint (github.com/AIM-Research-Lab/Medical-SAM3).
  *Verify first* that the Hugging Face weights are the 2D v1 model behind the paper's Table 3, not the later 3D "V2",
  and record the license (repo: SAM License; a Hugging Face snippet said MIT; unresolved).
- **Datasets:** external rows TN3K, HC18, PH2, CVC-ClinicDB and ETIS-Larib (required); CHASE_DB1 and STARE (optional).
- **Protocol:** the repo's 2D evaluation script, text prompts as fixed in code. Also report the GT-box numbers
  the script can produce but the paper omits.
- **Reproduction criterion (fix before running):** per-dataset Dice within ±3 points of Table 3 and the mean within ±2.
  Report misses with likely causes. A miss does not stop WP2-4, but it must be stated.
- **Reference models:** vanilla SAM 3 (same prompts); MedSAM as a box-prompt baseline if time allows.

## 3. WP2: Datasets, leakage and the zero-shot prompt sweep

**Modalities (fixed factor, 3 levels):**
| Modality | Role | Candidate dataset(s) | Status |
|---|---|---|---|
| Ultrasound | Held out | DDTI (thyroid) or BUS-BRA (breast) | Verify absent from Medical SAM3 Table 1. TN3K/SegThy (thyroid) and BUSI/BUS-UCLM (breast) are in training or external sets, so near-duplicate checks across them are required. DDTI has a loader in the repo, so check it isn't in V2 training |
| Dermoscopy | Held out | HAM10000 masks or IMA++ | High overlap risk with ISIC 2018, which is in training. Use cross-dataset dedup, and drop overlapping images |
| Endoscopy | **Seen-modality control** | Kvasir-SEG (in Medical SAM3 training) | Tests whether unlearning behaves differently when the clean pattern was already learned |

Avoid BrEaST, because UltraSAM3 already uses it externally.

**Leakage control.**
- Run group-wise splits per dataset with `src/dedup.py` and `src/make_splits.py`.
- Run a **cross-dataset** near-duplicate check of each held-out set against Medical SAM3's same-modality training
  sets (to be added to `dedup.py`). Any image matching a training image is dropped, not just regrouped.
- Kvasir-SEG lessons from the v1 run: hash-only linking produced false merges, and zoomed frames were missed.
  The linking rule must require image similarity and be zoom-tolerant (scale search or a pretrained embedding).
  Thresholds are calibrated on a hand-labelled set of ambiguous pairs before any training.

**Partitions per dataset** (group-wise; same proportions as v1): train pool 65% (corruption applied here),
trusted 10% (always clean), validation 10% (dev seed only), test 15%.

**Zero-shot prompt sweep (feeds the protocol, WP4).** Medical SAM3 and vanilla SAM 3, before any fine-tuning,
on each held-out test split:
- text: canonical term, plus 2 pre-registered synonyms;
- box: tight, plus jitter of 10/20/30%;
- point: 1/3/5 points, with shifts of 0/10/25% of the lesion radius;
- text+box.

**Point prompts:** the Medical SAM3 eval code has no point path. Use SAM 3's native interactive path. If it can't be
driven with the Medical SAM3 weights, drop points and report that explicitly. Do not substitute another model silently.

## 4. WP3: Conditions (randomized complete block; block = seed × modality)

Every condition starts from the **released Medical SAM3 checkpoint** and fine-tunes on the held-out dataset with
**text prompts only** (matching the paper's training regime). This makes an exact oracle cheap. Retraining
Medical SAM3 itself (33 datasets, 4× H100) is out of scope.

| Condition | Fine-tuned on | Role |
|---|---|---|
| Clean | All train-pool images, true masks | Audit reference (null) |
| Corrupted | Train pool, 40% with corrupted masks | Positive control |
| Oracle | Train pool minus **all** corrupted | Exact-unlearning target |
| Unlearned: SSD, SAP | Corrupted model + flagged subset (+ trusted set for SAP) | Post-hoc approximate unlearning, computed under text prompts |
| Unlearned: RmF | Fine-tune from the checkpoint minus **flagged** | Goel et al. baseline |

**Fine-tuning scope (fix before starting):** LoRA on the image encoder plus a fully trained mask/pixel decoder;
the presence head is trainable. If compute forces a frozen encoder, encoder-level audit claims are void.
**Check the SSD and SAP footprint:** SAM 3 is far larger than SAM ViT-B. Apply SSD (Fisher-based dampening) and
SAP (activation SVD) to the trainable parameters only, and record the choice.

**Reduced scale (v1 was budgeted for SAM ViT-B):**
- Flag fractions: **10, 50, 100%** (nested). RmF at 100% equals the Oracle, which serves as a pipeline check.
- SSD/SAP draws: **3** independent flagged-subset draws at 10% and 50%; 1 draw at 100%.
- Per seed × modality: 3 references + 2 RmF fine-tunes + 2 × (2 × 3 + 1) = 14 SSD/SAP models,
  so **5 fine-tunes and 16 unlearned models**.
- **Seeds:** 1 development (0) + **6 confirmatory (provisional, Sec. 10)**.
- **Totals:** 7 seeds × 3 modalities × 5 = **105 fine-tunes**, and 6 × 3 × 16 = **288 confirmatory unlearned models**.

If compute is short, cut in this order: drop the 10% fraction, then the Kvasir control, then reduce draws.
Never cut seeds below the power target.

**Corruption families.**
1. **Primary, synthetic:** an outer ring = dilate(mask, d + w) − dilate(mask, d), with w scaled to lesion size
   (e.g., 8% of the equivalent diameter, at the model's 1008 px input). Assignment is stratified by lesion-size tercile.
   Fix d and w on the dev seed.
2. **Secondary, real (exploratory):** IMA++ novice or flood-fill masks paired with expert masks of the same ISIC
   image, and BUSI duplicates with conflicting masks. Both overlap with Medical SAM3 training data (ISIC 2018, BUSI),
   so there is no clean oracle. Report descriptively only.

## 5. Recovery and prompt arms (test time)

**(a) Prompt arm.** No weight updates. Fully crossed within model and image, with fixed per-image perturbation seeds:
text (canonical and 2 synonyms), box (tight; expanded 10/20/30%, where expansion covers the ring), point (1/3/5;
shifts of 0/10/25%), and text+box. This is about 15 conditions, with 3 random draws per stochastic condition.
**Training prompt = text**, so text is the reference level for H4.

**(b) Fine-tuning arm.** Continued fine-tuning on the trusted set, with checkpoints at 0/100/300/1000 steps, run
identically on the Oracle.

**(c) Quantization arm (secondary, cheap).** 8-bit and 4-bit weight quantization of each unlearned model.
Literature shows that quantization can revive unlearned knowledge.

*Optional:* relearning on a few **unflagged** corrupted samples, timed against the Oracle.

## 6. The audit signal, controls and baselines

**Signals on the error band and on control bands:**
- pre-threshold mask logits;
- pixel-decoder features;
- image-encoder features (only if the encoder is adapted);
- the presence-head score, which is not spatial but is a candidate mediator for H4.

**Standardization:** A = (signal − Clean mean) / Clean SD, per modality. The Clean reference is fit on the
dev seed plus held-out Clean seeds only.

**Specificity controls (required, carried over from v1):**
- an offset band at a different distance;
- an inner band of the same width inside the true mask;
- a frozen-component sanity check, where A must be ≈ 0 for unadapted parts.

**Baselines the audit must beat** (all at the same level of analysis):
- output-level metrics M0 (EB-FP, Dice, mean ring logit);
- weight-space L2 distance to the Oracle and linear mode connectivity (Siddiqui et al.);
- a J-Access-style partial correlation controlling for output-level forgetting (Song et al.);
- global CKA/PCA shift;
- a linear probe for ring versus non-ring pixels on features;
- a segmentation membership-inference score.

**Never use A as a training or selection objective.** Song et al. show that optimizing an audit hides knowledge
from it and worsens recovery.

## 7. Pre-specified analysis

- **H0 (reproduction).** WP1 criterion (Sec. 2). Descriptive.
- **H1 (signal validity).** A_Corrupted > A_Oracle and A_Clean at the trained offset, and not at the control offset,
  in each modality. Model: A ~ condition × band × modality + (1|seed) + (1|image).
- **H2 (residual influence).** Some unlearned models are equivalent to the Oracle on EB-FP under text prompts
  (TOST, margin fixed on the dev seed), yet A is above the Oracle.
- **H3 (prediction, co-primary).** Among models that look clean, compare M0 against M0 + A for R in each recovery
  arm: likelihood-ratio test and ΔAUC. Run at **model level** (seed and modality as blocks) and at **pixel/image level**
  (R ~ A + M0 + (1|seed/model) + (1|image)). Report the baselines in Sec. 6 alongside.
- **H4 (prompt-conditional re-emergence, co-primary).** R ~ prompt type × modality + method + flag fraction
  + (1|seed/model) + (1|image), with text as the reference. Test box > text and point > text. Mediation by the
  presence score is secondary.
- **Multiplicity:** the H3 arms (prompt, fine-tune) and H4 make 3 co-primary tests, each at α = 0.05/3 ≈ 0.017
  (Bonferroni). The quantization arm and all modality-specific contrasts are secondary.

## 8. WP4: Contribution deliverables

1. **Prompt-selection protocol** per modality. Choose the prompt that maximizes the lower 95% CI of Dice, subject to
   (a) R after correction at or below the Oracle threshold, and (b) annotation cost (text < point < box).
   Ties go to the cheaper prompt. Derive it on validation data and evaluate it on test data only.
2. **Failure taxonomy.** Start from Medical SAM3's categories (near-empty mask, over-segmentation flood,
   thin-structure false positives, fragmented small targets, background activation) and add unlearning-specific ones:
   residual leakage, prompt-bypass re-emergence, over-forgetting (true lesion lost). Two raters label a stratified
   sample blind to condition; report Cohen's κ.
3. **Released code:** data prep (dedup, splits, corruption), the unlearning methods, the audit, and the protocol.
   Release no data, only scripts and checksums. Check license compatibility with the checkpoint's license first.

## 9. Fairness and blinding (unchanged from v1)

- Tune SSD/SAP/RmF on seed 0 only, with equal budgets, selecting by output metrics (validation Dice and EB-FP on
  flagged images under text), never by A.
- Freeze code, thresholds, bands, prompts, synonyms and the analysis script before seeds 1-6, and record the commit hash.
- Use a randomized execution order (regenerated `runs.csv`), and exclude the dev seed from confirmatory statistics.

## 10. Replication, power and compute: to redo

- **Power:** v1's simulation (`power_sim_reps.py`) gave 0.95-0.99 for H3 with 36 unlearned models per seed. v2 has
  16 per seed × modality, or 48 per seed pooled with modality blocks. Re-run the simulation with that structure and
  α = 0.017. H4 is not yet simulated: add a simulation for the prompt-type contrast, with image and model random
  effects. Re-estimate variance components on dev seed 0 before freezing the seed count.
- **Compute:** not measured. SAM 3 runs at 1008 px and is much larger than SAM ViT-B.
  - On seed 0, time one fine-tune, one SSD, one SAP and the full prompt arm for one model per modality.
  - Then extrapolate to 105 fine-tunes and 288 models × ~15 prompt conditions × 3 recovery arms.
  - Run **get-available-resources** before committing a GPU plan.
- **Regenerate:** `runs.csv` (seeds × modalities × conditions, randomized execution order), `prompt_conditions.csv`
  (with text, synonyms and text+box) and `recovery_finetune.csv`.

## 11. Verification before any "first to" claim

1. Read the full text of **SCU** (Du et al., AAAI 2026): its backbone, prompt types, feature-level and relearning analysis. This is the largest novelty risk.
2. Confirm the **checkpoint identity and license** (Sec. 2).
3. Confirm the **held-out datasets** are absent from Medical SAM3 Table 1, from UltraSAM3's external sets and, where possible, from SAM 3's pretraining; run the cross-dataset dedup.
4. Read the full texts of J-Access, DAMP, Erase to Retain, arXiv 2606.16180, RePAIR, the TRAK-for-segmentation submission, and arXiv 2608.30046.
5. Run a fresh search of arXiv and MICCAI 2026 for Sep-Nov 2026: "SAM 3 unlearning", "MedSAM unlearning", "prompt-type unlearning", "corrective unlearning segmentation".
6. **Pilot H1 on seed 0.** If fine-tuning doesn't learn the ring, there is nothing to unlearn. ADELE warns that structured noise may skip early learning.

## 12. Threats to validity

- **Synthetic corruption.** Handled by the real-error exploratory arm; claims stay limited to the ring family otherwise.
- **Boundary confound.** Handled by the offset and inner-band controls.
- **Leakage into Medical SAM3's training data.** Handled by cross-dataset dedup. Endoscopy is *declared* in-distribution and used as a control, never as transfer evidence.
- **Prompt as a confounder.** R is always computed against the Oracle under the same prompt.
- **Synonym choice.** Synonyms are fixed before seeing results; report results for each synonym.
- **Point-prompt feasibility.** If unavailable, H4 reduces to box vs text, and the protocol is reported without points.
- **Single base model.** Results are about Medical SAM3; vanilla SAM 3 is a reference, not a replication.
