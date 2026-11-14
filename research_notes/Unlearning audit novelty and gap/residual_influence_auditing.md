# Residual-Influence Auditing of Machine Unlearning: Representation-Level Audits, Relearning/Recovery, and Spatial Attribution (state as of Nov 2026)

Method note: All findings come from web searches run on 2026-11-14. Direct fetches of arxiv.org, alphaxiv.org and openreview.net failed because those hosts are blocked in this session's network, so most details come from search-result excerpts of arXiv HTML/PDF pages, conference pages and paper aggregators, not from full-text reading. Treat exact numbers as "reported in excerpts" and check them against the PDFs before citing. Items marked **[unverified this session]** are known from prior knowledge but were not confirmed by a search here.

## Q1. Representation-level unlearning evaluation (probes, CKA, SAE restoration, "suppression not deletion"). Which are spatial or dense-prediction?

### Takeaway
Many 2024-2026 papers now show that output-level forgetting often hides information that survives in representations: SAE restoration, linear probes, CKA/PCA drift, information decomposition and Jacobian/logit lenses. All of them work on whole samples or whole classes (classification, LLM tokens, VLM entities). None of the representation-level audits found here measures residual signal at specific spatial locations or pixels in a dense-prediction model.

### Cited Findings
**The team's anchor paper**
- Jang, Lee, Kim, Jo, Woo, "Suppression or Deletion: A Restoration-Based Representation-Level Analysis of Machine Unlearning" (arXiv 2602.18505, submitted 18 Feb 2026, WWW '26 short paper). The paper defines *deletion* as removing class-specific representations from all layers and *suppression* as those representations staying encoded in intermediate layers while being masked at the output. It finds class-specific features with Sparse Autoencoders, then uses inference-time steering to test restoration. Across 12 unlearning methods on image classification, most show high restoration rates. Even retraining from pretrained checkpoints shows high restoration. Code is at github.com/Yurim990507/suppression-or-deletion. — [arXiv HTML](https://arxiv.org/html/2602.18505v1); [GitHub mirror](https://gh.r-mj.com/Yurim990507/suppression-or-deletion)

**LLM representation-level audits**
- Xu et al., "Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs" (arXiv 2505.16831; ICML 2026 poster). It uses four representation diagnostics: PCA similarity, PCA shift, CKA and Fisher information. It describes *reversible* forgetting (token metrics collapse while PCA-Sim and CKA stay high) and *irreversible* forgetting (low PCA-Sim/CKA, large PCA shift, persistent Fisher shift). The paper notes that task scores alone cannot tell the regimes apart, and the difference only shows after relearning. Toolkit: github.com/XiaoyuXU1/Representational_Analysis_Tools. — [arXiv v2](https://arxiv.org/html/2505.16831v2); [ICML 2026](https://icml.cc/virtual/2026/poster/65395)
- Goel, Ritter, Gurevych, "Auditing Language Model Unlearning via Information Decomposition" (arXiv 2601.15111). It uses Partial Information Decomposition between pre- and post-unlearning representations to separate the "residual knowledge" that survives from the information that is actually removed. — [arXiv](https://arxiv.org/abs/2601.15111)
- "Does Machine Unlearning Truly Remove Model Knowledge? A Framework for Auditing Unlearning in LLMs" (arXiv 2505.23270). It perturbs intermediate activations at inference time to test whether the forgotten target still shapes outputs. This is a steering-style audit like Jang et al., but for LLMs. — [arXiv HTML](https://arxiv.org/html/2505.23270v1)
- "REMIND: Input Loss Landscapes Reveal Residual Memorization in Post-Unlearning LLMs" (arXiv 2511.04228). It argues that single-input checks miss residual influence and looks at loss over small input neighborhoods. This is conceptually close to the team's idea of testing under prompt perturbation. — [arXiv HTML](https://arxiv.org/html/2511.04228)
- "Robust LLM Unlearning Against Relearning Attacks: The Minor Components in Representations Matter" (arXiv 2605.11685). It reports that output-level unlearning leaves internal representations essentially intact and trivially recoverable, and defines a recovery ratio (about 1 means full reversal, about 0 means robust). — [arXiv HTML](https://arxiv.org/html/2605.11685v1)
- Hong et al., "Intrinsic Evaluation of Unlearning Using Parametric Knowledge Traces" (ConceptVectors; arXiv 2406.11614) **[unverified this session]**. It locates concept vectors in MLP parameters and shows that behavioral unlearning leaves them largely intact.

**Vision (classification) representation-level audits**
- Jung et al., "OPC: One-Point Contraction" (arXiv 2507.07754). It argues that even when logit metrics show successful unlearning, features keep enough structure for classification recovery and for image reconstruction through external decoders. It proposes contracting forget-set features to a single point. — [arXiv HTML](https://arxiv.org/html/2507.07754v1)
- "Mirage" (arXiv 2605.20282). It reports that linear probes recover forgotten-label information with high accuracy and calls this gap the "forgetting illusion". — [arXiv](https://arxiv.org/pdf/2605.20282)
- POUR (arXiv 2511.19339) and Lee et al. (arXiv 2602.05375) also report that linear probing recovers substantial forget-set accuracy. — [POUR](https://arxiv.org/html/2511.19339v2); [2602.05375](https://arxiv.org/pdf/2602.05375v2)
- A search excerpt describes an "FM-recovery" diagnostic: a single linear map fitted on a held-out calibration set, without forget data, reverses 14 unlearning methods on CIFAR-10/SVHN in seconds. The excerpt attributed it to 2507.07754, but that attribution looks doubtful and the source paper is **uncertain**. Verify before citing. — [search excerpt source](https://arxiv.org/pdf/2507.07754)
- "Gradient Concentration, Not Weight Saliency, Explains Representation-Level Class Unlearning" (arXiv 2607.21353). It uses a linear probe and a few-shot prototype attack in penultimate feature space to test whether the forgotten class is still recoverable. — [arXiv](https://arxiv.org/pdf/2607.21353)
- "Classification-Head Bias in Class-Level Machine Unlearning" (arXiv 2605.08730). It argues that retain and forget accuracy cannot tell real removal of internal knowledge from shortcuts in the final layer. — [arXiv](https://arxiv.org/pdf/2605.08730)
- "Unlearning's Blind Spots: Over-Unlearning and Prototypical Relearning Attack" (arXiv 2506.01318). — [arXiv](https://arxiv.org/pdf/2506.01318)

**Dense-prediction / image-output evaluation (closest to spatial)**
- RUB, "Evaluating Residual Knowledge in Unlearned Models" (arXiv 2504.14798). An aggregator review says that for image-to-image models it uses per-pixel L1 distance between unlearned and original outputs to measure restored content. This is an *output-space* pixel metric, not an internal-signal audit. — [Pith review](https://pith.science/paper/2504.14798)
- "Realistic Image-to-Image Machine Unlearning via Decoupling and Knowledge Retention" (arXiv 2502.04260). A review describes a poisoning/backdoor-based audit for image-to-image unlearning. — [Pith review](https://pith.science/paper/2502.04260)
- DAMP, "Class Unlearning via Depth-Aware Removal of Forget-Specific Directions" (arXiv 2604.15166). It includes semantic segmentation (Pascal VOC 2012, DeepLabV3-ResNet50, forgetting "Human"). It compares qualitatively against retrain, GA, KD, logit masking, random relabel, SSD and SalUn, and reports that DAMP stays closer to the retrained model by suppressing forgotten regions. This is the closest *dense* comparison to a retrained reference, but it is qualitative/output-level and not a per-pixel internal audit. — [arXiv](https://arxiv.org/pdf/2604.15166)
- ADVR (GitHub repository only, not peer-reviewed). It proposes "adversarial-replay class-unlearning for semantic segmentation" and argues for removing the spatial "silhouette" of a forgotten class, not just relabeling its pixels. Unverified claims, but the idea is close to "spatial residue". — [GitHub](https://github.com/KAVYANSHTYAGI/advr-unlearning)

### Inferences
- Reviewers will see Jang et al., Xu et al. (2505.16831), OPC and the linear-probe papers as the representation-level baselines. The team's spatial audit is effectively "Jang/Xu-style residual detection, localized to the corrupted pixels, with a clean-model reference". The reference to a clean-trained model is itself standard (the retrain gold standard).
- The team's per-pixel internal-signal audit (feature maps or decoder logits at the false-ring pixels compared with a clean reference) has no direct counterpart in what was found. The nearest are output-space per-pixel L1 (RUB) and qualitative forgotten-region maps (DAMP).

### Gaps
- No full-text check of DAMP's segmentation evaluation was possible (arxiv.org blocked). It may contain per-region quantitative metrics.
- Could not confirm which paper introduced "FM-recovery".
- Did not find CKA or probe analyses computed *per spatial location* in any unlearning paper. This absence comes from searching, not proof that none exist.

## Q2. Relearning/recovery attacks, and whether any pre-perturbation metric predicts recovery

### Takeaway
Recovery after benign or targeted fine-tuning, quantization and prompting is well established in LLMs, diffusion models, VLMs and small vision classifiers. "A pre-attack audit predicts recovery" is **already published**:
- Song et al.'s J-Access (arXiv 2608.11408, LLMs) shows a model-level correlation with recovery but no item-level prediction.
- Siddiqui et al. (NeurIPS 2025, vision classifiers) show that weight-space L2 distance and linear mode connectivity track resistance to relearning.
- Xu et al. use representation metrics to separate reversible from irreversible regimes.

None of these does it spatially or for segmentation.

### Cited Findings
**Pre-perturbation predictors of recovery (most important prior art for "audit predicts re-emergence")**
- Song et al. (11 authors), "Measure, Don't Optimize: Forecasting Recovery in LLM Unlearning" (arXiv 2608.11408, Aug 2026):
  - **What J-Access is:** an inference-time audit that maps intermediate representations to vocabulary space with a Jacobian lens and measures how often target concepts stay accessible along the output pathway.
  - **Hypothesis:** knowledge closer to the output pathway needs less fine-tuning to restore.
  - **Scale:** it audits 398 OpenUnlearning models and compares against TOFU answer probability, six MIAs and a logit-lens baseline.
  - **Model-level results:** higher pre-attack J-Access goes with greater recovery. The partial correlation stays about +0.33 after controlling for behavioral forgetting and utility metrics, and the relation holds within method families. Most unlearned models keep access above the retain-only gold level.
  - **No item-level prediction:** the metric cannot identify *which* facts will be recovered, and the authors say it gives no item-level deletion certificate.
  - **Gaming:** directly minimizing J-Access makes the model hide knowledge from the audit, with lower audit scores but greater post-attack recovery.
  - Sources: [arXiv PDF](https://arxiv.org/pdf/2608.11408); [summary page](https://www.roboticscenter.ai/research/papers/measure-dont-optimize-forecasting-recovery-in-llm-unlearning-2608)
- Siddiqui, Weller, Krueger, Dziugaite, Mozer, Triantafillou, "From Dormant to Deleted: Tamper-Resistant Unlearning Through Weight-Space Regularization" (arXiv 2505.22310; NeurIPS 2025):
  - **Setting:** forgetting individual examples in small vision classifiers.
  - **Recovery:** forget-set accuracy rises from about 50% after unlearning to nearly 100% after fine-tuning on the retain set only. A model retrained from scratch stays at about 50%.
  - **Predictors:** resistance to relearning tracks the L2 distance between the original and unlearned weights and the linear mode connectivity between them.
  - **Defense:** the authors propose weight-distortion methods that increase weight-space separation.
  - Sources: [arXiv HTML](https://arxiv.org/html/2505.22310v2); [NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/bbeafbaa60ee03f2c3135be71ebd5d06-Abstract-Conference.html)
- Xu et al. (arXiv 2505.16831). Representation metrics (PCA-Sim, CKA, PCA shift, FIM) separate reversible from irreversible forgetting regimes, which they label using relearning. — [arXiv v2](https://arxiv.org/html/2505.16831v2)
- Che, Casper et al., "Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities" (arXiv 2502.05209; TMLR 2025):
  - **Benchmark:** 8 unlearning methods and 9 safety-tuned LLMs, each tested against 11 elicitation attacks.
  - **Fragility:** state-of-the-art unlearning is undone within 16 fine-tuning steps.
  - **Prediction:** few-shot fine-tuning attack success conservatively over-estimates vulnerability to held-out input-space attacks. In other words, one perturbation predicts another, which is relevant to the team's "fine-tuning versus prompt-perturbation" pairing.
  - **Release:** 64 unlearned models (LLM-GAT on Hugging Face).
  - Sources: [arXiv v4](https://arxiv.org/html/2502.05209v4); [TMLR](https://mlanthology.org/tmlr/2025/che2025tmlr-model/)
- Suriyakumar et al., "Unstable Unlearning: The Hidden Risk of Concept Resurgence in Diffusion Models" (arXiv 2410.08074). Per the search summary, it argues theoretically that projection overlap between the forgotten subspace and the fine-tuning gradient directions governs how strongly a concept comes back. This is a mechanistic predictor of resurgence. — [arXiv PDF](https://arxiv.org/pdf/2410.08074)

**Relearning / benign fine-tuning recovery**
- Hu, Fu, Wu, Smith, "Jogging the Memory of Unlearned LLMs Through Targeted Relearning Attacks" (arXiv 2406.13356; ICML 2024 FM-Wild workshop; NeurIPS 2024 listing). Small, loosely related relearning sets recover unlearned content: public medical articles bring back bioweapon knowledge, and general Harry Potter wiki text brings back verbatim memorized text. Relearning works better when the data is closer to the unlearned content. — [arXiv v2](https://arxiv.org/html/2406.13356v2); [NeurIPS](https://neurips.cc/virtual/2024/106237)
- Suriyakumar et al. (arXiv 2410.08074). Fine-tuning an unlearned Stable Diffusion model on *unrelated, non-adversarial* images causes concept resurgence. The summary reports about 16% resurgence with 50 samples and 1000 LoRA steps. — [arXiv PDF](https://arxiv.org/pdf/2410.08074)
- "Erased or Dormant? Rethinking Concept Erasure Through Reversibility" (arXiv 2505.16174). Erased concepts are recovered with high fidelity after minimal adaptation across six erasure algorithms. — [arXiv HTML](https://arxiv.org/html/2505.16174v2)
- "Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures" (arXiv 2602.03379). — [arXiv HTML](https://arxiv.org/html/2602.03379v1)
- "Memory Self-Regeneration" (arXiv 2510.03263), on recovery in diffusion unlearning. — [arXiv HTML](https://arxiv.org/html/2510.03263v2)
- Projected Gradient Unlearning against concept revival (arXiv 2604.21041), a defense. — [arXiv](https://arxiv.org/pdf/2604.21041)
- Fan et al., SAM-based (sharpness-aware) LLM unlearning resilient to relearning (arXiv 2502.05374, ICML 2025) **[unverified this session]**.

**Quantization recovery**
- Zhang et al., "Catastrophic Failure of LLM Unlearning via Quantization" (arXiv 2410.16454). Unlearned models keep about 21% of target knowledge at full precision and about 83% after 4-bit quantization, while 8-bit roughly preserves unlearning. The proposed explanation is that small unlearning weight changes get masked by 4-bit rounding. — [arXiv PDF](https://arxiv.org/pdf/2410.16454)
- Follow-ups: quantization-robust unlearning via LoRA (arXiv 2602.13151) and via retain-forget loss-landscape interaction (arXiv 2609.27355). — [2602.13151](https://arxiv.org/pdf/2602.13151); [2609.27355](https://arxiv.org/html/2609.27355v1)

**Corrupted / poisoned training data (closest to the team's corrupted-mask setup)**
- Goel, Prabhu, Torr, Kumaraguru, Sanyal, "Corrective Machine Unlearning" (arXiv 2402.14015; TMLR Oct 2024). It defines unlearning manipulated or incorrect training data when only a subset is identified. Even knowing 80% of the manipulated data is not enough for retraining-without-deleted-data (RewoD), or other methods, to remove effects of manipulating 1% of the data. SSD has limited success and fails completely on Interclass Confusion. Benchmark code: github.com/drimpossible/corrective-unlearning-bench. — [arXiv v2](https://arxiv.org/html/2402.14015v2)
- Pawelczyk et al., "Machine Unlearning Fails to Remove Data Poisoning Attacks" (arXiv 2406.17216, ICLR 2025) **[unverified this session]**.

### Inferences
- The team cannot claim that "a residual audit predicts relearning" is new in general. J-Access is the direct precedent, with partial-correlation controls the team should copy. Siddiqui et al. is the vision precedent, with weight-space L2 and LMC as cheap baselines to beat. What the team *can* claim is **spatial/item-level** prediction (does residue at pixel p predict re-emergence at pixel p?). J-Access explicitly says it cannot predict at the item level, which makes this a gap worth stating directly.
- J-Access's "measure, don't optimize" finding warns that the team should not also use the audit as an unlearning objective without testing whether it gets gamed.
- The team's setup (corrupted masks, a known artifact, unlearning) is an instance of Goel et al.'s corrective unlearning. Reviewers will expect that paper to be cited and its "partial identification of corrupted samples" condition to be considered.

### Gaps
- Could not read the J-Access full text: details such as the attacks used (fine-tuning only, or also quantization) and the exact correlation statistic come from excerpts.
- No paper found that tests whether an audit predicts *both* prompt-driven and fine-tuning-driven re-emergence in the same model, apart from Che et al.'s cross-attack prediction in LLMs.

## Q3. Influence/attribution for segmentation, and memorization of label noise in segmentation

### Takeaway
Training-data attribution for segmentation is thin. One OpenReview paper adapts TRAK to segmentation with per-pixel and per-class attribution targets, and "Training Feature Attribution" (arXiv 2510.09135) produces pixel-wise influence maps over training images but only for classification. The label-noise literature (ADELE) shows that segmentation networks fit clean pixels first and then memorize noisy ones on class-dependent schedules. This supports the premise that a false ring gets memorized, but none of this work connects attribution to unlearning audits.

### Cited Findings
- "Data Attribution for Segmentation Models" (anonymous OpenReview submission; venue and year **unknown**, full text not fetchable here). Per the search excerpt, it adapts TRAK to segmentation, argues that the attribution target is ambiguous for dense prediction, and proposes attributing per-pixel binary decisions (pixel belongs to class c). It compares full-image and per-class targets, and counterfactual retraining suggests that full-image TRAK scores find training examples that matter. — [OpenReview PDF](https://openreview.net/pdf/e15a148b35fbb56da1af7d88c7c66add68615a6a.pdf)
- "Training Feature Attribution for Vision Models" (arXiv 2510.09135). It combines training-data attribution with feature attribution: pixel-wise influence maps over *training* images, built by measuring how perturbing training pixels changes attribution to a given test prediction. Experiments use Pascal VOC 2012 with ResNet-18 at 224x224 and are classification, not segmentation. — [arXiv HTML](https://arxiv.org/html/2510.09135)
- Liu et al., "Adaptive Early-Learning Correction for Segmentation from Noisy Annotations" (ADELE; arXiv 2110.03740; CVPR 2022):
  - **Finding:** segmentation networks fit clean pixel labels first and then memorize wrong ones, and memorization onset differs by category.
  - **Method:** ADELE tracks per-class IoU curves to detect onset and corrects labels with the model's own outputs, plus a multiscale consistency term.
  - **Evaluation:** SegTHOR with synthetic annotation-like noise, and PASCAL VOC weak supervision.
  - **Limitation:** when noise is highly structured, early learning may not happen. A *systematic* false ring is exactly such structured noise.
  - Sources: [arXiv PDF](https://arxiv.org/pdf/2110.03740); [CVPR supplement](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Liu_Adaptive_Early-Learning_Correction_CVPR_2022_supplemental.pdf)
- "A Bayesian Approach to Segmentation with Noisy Labels via Spatially Correlated Distributions" (arXiv 2504.14795), on spatially correlated label noise modeling. — [arXiv](https://arxiv.org/pdf/2504.14795)
- Medical noisy-label segmentation correction (not unlearning):
  - Clean Label Disentangling — [arXiv 2311.16580](https://arxiv.org/pdf/2311.16580)
  - Adaptive Label Correction — [arXiv 2503.12218](https://arxiv.org/abs/2503.12218)
  - SAM-Correction — [arXiv 2610.32377](https://arxiv.org/abs/2610.32377)
- Noisy-label unlearning (classification-oriented):
  - "Forget or Fine-tune? A Comparative Study of Machine Unlearning Strategies for Noisy Label Correction" (arXiv 2608.30046). It frames noisy samples found by filtering as the forget set, with retraining on clean data as the reference. Findings could not be retrieved. — [arXiv](https://arxiv.org/pdf/2608.30046)
  - COLUR (IJCAI 2025, arXiv 2506.19496) — [arXiv](https://www.arxiv.org/pdf/2506.19496)
  - CUTS, corrective unlearning via task arithmetic without the forget set (arXiv 2511.18660) — [arXiv](https://web3.arxiv.org/pdf/2511.18660v1)
  - He et al., "Towards Natural Machine Unlearning" (arXiv 2405.15495) — [arXiv](https://arxiv.org/pdf/2405.15495)
- Georgiev et al., "The Journey, Not the Destination" (diffusion TRAK, arXiv 2312.06205) **[unverified this session]**. It is a possible reference for attribution in models with spatial outputs.

### Inferences
- The OpenReview TRAK-for-segmentation paper is the natural attribution baseline: per-pixel TRAK targets restricted to the ring pixels. The team could show that their cheap internal-signal audit agrees with, or is cheaper than, per-pixel TRAK scores toward the corrupted training masks.
- The ADELE limitation (structured noise may skip the early-learning phase) suggests that the false ring may be learned "as signal". That would make it easier to suppress than to delete, which strengthens the motivation for a spatial residual audit.

### Gaps
- Could not verify the authors, venue or acceptance status of "Data Attribution for Segmentation Models".
- No influence-function or datamodel work found that is designed for *unlearning verification* in segmentation.

## Q4. Prompt-dependent re-emergence in promptable models (SAM, VLMs, LLMs, diffusion)

### Takeaway
Re-emergence of forgotten behavior under prompt changes is well documented: concept-erasure circumvention in diffusion, in-context attacks on unlearned VLMs, and prompt emphasis in LLMs. **No work was found on unlearning in SAM or other promptable segmentation models, or on prompt-perturbation recovery there.** The team's SAM/prompt-perturbation angle appears open.

### Cited Findings
- "On the Robustness of Machine Unlearning for Vision-Language Models" (arXiv 2605.26992, May 2026). It uses three attacks: in-context (cues added at inference, no weight change), in-distribution fine-tuning on forget data, and out-of-distribution fine-tuning. Many VLM unlearning methods stay vulnerable, which suggests they *conceal* rather than remove knowledge. It argues that standard-prompt evaluation overstates removal. Code: github.com/XMUDeepLIT/VLM-UnL-Attack. — [arXiv HTML](https://arxiv.org/html/2605.26992); [HF papers](https://huggingface.co/papers/2605.26992)
- "Robust MLLM Unlearning via Visual Knowledge Distillation" (arXiv 2512.11325). — [arXiv](https://arxiv.org/pdf/2512.11325)
- Diffusion concept erasure:
  - A search summary says Pham et al. (2024) revived erased concepts with adversarial prompts (Concept Inversion, "Circumventing Concept Erasure Methods for Text-to-Image Generative Models", arXiv 2308.01508, ICLR 2024 **[ID unverified this session]**), and Lu et al. (2025) circumvented erasure via prompt perturbation, inpainting and noise probing. — [summary context in arXiv 2410.08074](https://arxiv.org/pdf/2410.08074)
  - UnlearnDiffAtk (Zhang et al., arXiv 2310.11868) and Ring-A-Bell (arXiv 2310.10012) are adversarial-prompt attacks on unlearned diffusion models **[unverified this session]**.
- LLMs: search excerpts report that emphasizing key tokens in a prompt can bring back unlearned knowledge, with target knowledge often not truly erased (arXiv 2608.29943, "On the Recoverability of Private Information Unlearning in LLMs"; attribution of the specific claim to this paper is from a snippet). — [arXiv](https://arxiv.org/pdf/2608.29943)
- Searches for SAM unlearning ("Segment Anything SAM machine unlearning prompt") returned only SAM adaptation papers (for example Polyp-SAM, arXiv 2305.00293), with nothing on unlearning or forgetting. — [Polyp-SAM](https://arxiv.org/pdf/2305.00293)

### Inferences
- Reviewers would cite the diffusion concept-erasure circumvention line and the VLM robustness paper (2605.26992) as prior art for "forgotten behavior returns under prompt changes". The specific claim, that a false-ring artifact returns in a *promptable segmentation* model when point/box prompts are perturbed, appears unpublished.
- Che et al.'s finding that fine-tuning attacks over-estimate input-space vulnerability gives a hypothesis to test spatially: fine-tuning re-emergence should upper-bound prompt-perturbation re-emergence at the ring pixels.

### Gaps
- One general search for SAM unlearning may have missed niche workshop papers. Absence of evidence is not conclusive.

## Q5. Membership inference and verification of unlearning for segmentation models

### Takeaway
MIAs on segmentation exist (ECCV 2020; arXiv 2212.01082), and segmentation unlearning methods and benchmarks began appearing in 2025-2026, mostly in medical imaging. Their evaluations are forget/retain IoU or Dice, plus MIA in some cases. No segmentation unlearning paper found uses internal, spatially resolved residual audits or relearning-prediction tests.

### Cited Findings
- He et al., "Segmentations-Leak: Membership Inference Attacks and Defenses in Semantic Image Segmentation" (ECCV 2020). This is the first MIA study on segmentation (Cityscapes, BDD100K, Mapillary Vistas), and its defenses had little effect on segmentation quality. — [ECCV page](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4364_ECCV_2020_paper.php)
- Chobola, Usynin, Kaissis, "Membership Inference Attacks Against Semantic Segmentation Models" (arXiv 2212.01082). It covers binary segmentation and model poisoning. Segmentation can be considerably more vulnerable than classification under some threat models, and only DP-SGD protected membership. — [arXiv PDF](https://arxiv.org/pdf/2212.01082)
- "To forget is to preserve: Machine Unlearning for 3D medical image segmentation" (arXiv 2606.16180, 2026). It benchmarks approximate unlearning on MRBrainS18 with subject-level forget sets and voxel-wise outputs. A "Noisy Label" strategy (random voxel relabeling) gave the best trade-off: 93% decrease on the forget set and 84% retain accuracy after 50 epochs. — [arXiv HTML](https://arxiv.org/html/2606.16180v1)
- Datta and Alam, "Erase to Retain: Low Rank Adaptation Guided Selective Unlearning in Medical Segmentation Networks" (arXiv 2511.16574, 20 Nov 2025):
  - **Method:** LoRA teacher-student with a frozen backbone, strong-unlearn then restore phases, and a background-driven forget loss (all-background mask on the forget set).
  - **Results (ISIC):** forget IoU 0.875 to 0.509, with retain/val IoU about 0.647-0.677 and about 2.9% of parameters updated. It also reports CHASE vessel results.
  - **Novelty claim:** the authors say selective unlearning for medical segmentation had not been examined before.
  - Numbers come from secondary summaries.
  - Sources: [arXiv](https://arxiv.org/pdf/2511.16574); [review](https://www.themoonlight.io/review/erase-to-retain-low-rank-adaptation-guided-selective-unlearning-in-medical-segmentation-networks)
- DAMP (arXiv 2604.15166) segmentation class-unlearning experiments: see Q1. — [arXiv](https://arxiv.org/pdf/2604.15166)
- The machine unlearning survey (arXiv 2405.07406) classifies "model-difference" audits (comparing models before and after unlearning) as a distinct verification category. — [arXiv](https://arxiv.org/pdf/2405.07406)

### Inferences
- Segmentation unlearning is an emerging area (2025-2026) with output-level metrics only. A team audit that adds (a) internal per-pixel residue against a clean reference and (b) validated prediction of re-emergence would add evaluation depth these papers lack.
- An MIA baseline adapted from Segmentations-Leak/Chobola is a sensible "black-box verification" comparator. The team could show that pixel-level internal residue detects incomplete unlearning when MIA does not.

### Gaps
- Could not confirm whether 2606.16180 or 2511.16574 include MIA or relearning evaluations (full text not accessible).

## Verdict: novelty and likely prior art

### Takeaway
**Likely novel:**
- (1) A *spatially localized* residual-influence audit: internal signals at the exact corrupted pixels compared with a clean-trained reference, for corrective unlearning in *segmentation*.
- (2) Testing whether that *pixel/item-level* audit predicts *where* an artifact re-emerges.
- (3) Prompt-perturbation re-emergence in promptable segmentation models such as SAM, if the team uses one.

**Not novel:**
- "Unlearning is suppression, not deletion".
- Representation-level restoration audits.
- Relearning, benign fine-tuning and quantization recovery.
- "A pre-attack audit predicts recovery" at the model level (J-Access; Siddiqui et al.).
- Corrective unlearning of corrupted labels (Goel et al.).

### Cited Findings (prior art reviewers will likely cite)
- Suppression vs deletion, representation-level audits:
  - Jang et al. 2602.18505 — [link](https://arxiv.org/html/2602.18505v1)
  - Xu et al. 2505.16831 — [link](https://arxiv.org/html/2505.16831v2)
  - OPC 2507.07754 — [link](https://arxiv.org/html/2507.07754v1)
  - Mirage 2605.20282 — [link](https://arxiv.org/pdf/2605.20282)
  - Goel/Ritter/Gurevych PID audit 2601.15111 — [link](https://arxiv.org/abs/2601.15111)
- Audit predicts recovery:
  - J-Access 2608.11408 (model-level only, explicitly not item-level) — [link](https://arxiv.org/pdf/2608.11408)
  - Siddiqui et al. 2505.22310 (weight-space L2/LMC, vision) — [link](https://arxiv.org/html/2505.22310v2)
  - Che et al. 2502.05209 (fine-tuning attacks predict input-space vulnerability) — [link](https://arxiv.org/html/2502.05209v4)
- Recovery phenomena:
  - Hu et al. 2406.13356 — [link](https://arxiv.org/html/2406.13356v2)
  - Suriyakumar et al. 2410.08074 — [link](https://arxiv.org/pdf/2410.08074)
  - Zhang et al. 2410.16454 — [link](https://arxiv.org/pdf/2410.16454)
  - VLM robustness 2605.26992 — [link](https://arxiv.org/html/2605.26992)
- Corrective unlearning setup: Goel et al. 2402.14015 — [link](https://arxiv.org/html/2402.14015v2)
- Segmentation unlearning:
  - DAMP 2604.15166 — [link](https://arxiv.org/pdf/2604.15166)
  - Erase to Retain 2511.16574 — [link](https://arxiv.org/pdf/2511.16574)
  - 3D medical seg unlearning 2606.16180 — [link](https://arxiv.org/html/2606.16180v1)
- Spatial attribution and noise memorization:
  - TRAK-for-segmentation (OpenReview) — [link](https://openreview.net/pdf/e15a148b35fbb56da1af7d88c7c66add68615a6a.pdf)
  - Training Feature Attribution 2510.09135 — [link](https://arxiv.org/html/2510.09135)
  - ADELE 2110.03740 — [link](https://arxiv.org/pdf/2110.03740)
  - RUB 2504.14798 (per-pixel output L1) — [link](https://pith.science/paper/2504.14798)

### Inferences (recommended evaluation methods to adopt or compare against)
1. **Gold reference:** a clean-trained model, retrained on corrected masks, as the reference for every audit. This is standard and expected; see Goel et al. for the partial-identification variant, where only some corrupted masks are known.
2. **Baselines for the spatial audit:**
   - Global CKA/PCA-shift (Xu et al.).
   - Linear probes and k-NN on encoder/decoder features (Mirage/OPC-style), but trained to predict "ring vs. non-ring" *per pixel*.
   - SAE-feature restoration via steering (Jang et al.), applied to decoder feature maps.
   - A logit-lens/Jacobian-lens analog (J-Access): project intermediate feature maps through the mask head and measure ring-pixel logit mass.
   - Weight-space L2 and linear mode connectivity to the corrupted model (Siddiqui et al.), as cheap non-spatial predictors to beat.
   - Output-space per-pixel L1/Dice on the ring band (RUB/DAMP-style).
   - MIA on segmentation (Chobola et al.).
3. **Recovery attacks to run:**
   - Retain-only benign fine-tuning (Siddiqui).
   - Loosely related data relearning (Hu).
   - Few-step fine-tuning (Che, 16 steps).
   - 4-bit/8-bit quantization (Zhang).
   - Prompt perturbation: point/box jitter for SAM-like models, in the spirit of the in-context and concept-inversion attacks.
4. **Statistics to copy from J-Access:**
   - Report correlation between the pre-attack audit and post-attack ring re-emergence *controlling for output-level forgetting metrics* (partial correlation).
   - Test within method families.
   - Report both model-level and item/pixel-level prediction, since the item level is the gap J-Access leaves open.
   - Include a "measure, don't optimize" check: if the audit is used as a loss, does recovery get worse?
5. **Framing for reviewers:** position the contribution as extending representation-level unlearning auditing to dense prediction with spatial localization. The headline claim should be the *item/pixel-level* predictive validity of the audit for re-emergence, which J-Access says its own method cannot provide. Do not claim novelty for "audit predicts relearning" in general.

### Gaps
- Full-text verification of J-Access, DAMP, Erase to Retain, 2606.16180 and the TRAK-for-segmentation paper was blocked (arxiv.org, alphaxiv.org and openreview.net are not reachable from this session). Allowing these hosts would enable checking exact metrics and whether any of them already include spatial residual analysis.
- Items marked [unverified this session] (ConceptVectors 2406.11614, Pawelczyk 2406.17216, Fan et al. 2502.05374, Pham et al. 2308.01508, UnlearnDiffAtk 2310.11868, Ring-A-Bell 2310.10012, Georgiev et al. 2312.06205) should be confirmed before citing.
- The novelty verdict rests on searches finding no spatial or pixel-level internal residual audit for segmentation unlearning. A final check of CVPR/ICCV/MICCAI 2026 proceedings, which may not be fully indexed, is advisable.
