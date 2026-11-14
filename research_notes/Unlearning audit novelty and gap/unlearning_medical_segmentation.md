# Machine Unlearning in Medical Image Segmentation and Segmentation Foundation Models (2023 to Nov 2026): Novelty Check for "Did It Really Forget?"

Method note: WebFetch could not reach arxiv.org, alphaxiv.org or ojs.aaai.org from this session because those hosts are blocked by the network settings. Every finding below comes from search-engine result content: abstracts, snippets and aggregator summaries. I read no full texts. Paper existence (title, arXiv ID or DOI, authors) is confirmed by several independent index pages. Method and evaluation details marked "(snippet)" should be checked against the PDF before the team cites them.

## Q1. Which papers do machine unlearning for medical image segmentation, of any architecture?

### Takeaway
Very few papers exist, and all appeared from late 2025 onward. The two direct ones are Erase to Retain (arXiv 2511.16574, Nov 2025) and To forget is to preserve (arXiv 2606.16180, Jun 2026). Both target privacy-style forgetting of samples, lesions or subjects. Both evaluate only outputs (IoU, Dice, MAE on the forget and retain sets). Neither uses SAM, neither deals with corrupted or noisy masks, and neither audits internal signals or re-emergence.

### Cited Findings

**Erase to Retain: Low Rank Adaptation Guided Selective Unlearning in Medical Segmentation Networks.** arXiv:2511.16574. Datta and Alam. Covered in detail under Q2.
- Its authors claim that "selective unlearning for medical segmentation has not been previously examined." — [arXiv 2511.16574](https://arxiv.org/abs/2511.16574)

**To forget is to preserve: Machine Unlearning for 3D medical image segmentation.** arXiv:2606.16180, June 2026, cs.CV with cs.LG cross-list.
- Authors: Nitesh Kumar Singh, Akhilesh Singh and Arjun Arora, School of Computer Science, UPES, Dehradun, India. — [arXiv 2606.16180](https://arxiv.org/abs/2606.16180)
- It presents itself as "the first systematic benchmark of multiple approximate machine unlearning strategies for 3D medical image segmentation using subject-level forget requests." — [arXiv html](https://arxiv.org/html/2606.16180)
- Model and data: a 3D ResNet-50 segmentation backbone pretrained with Med3D, on MRBrainS18. A third-party summary describes 10 subjects split into 2 forget and 8 retain. — [arXiv 2606.16180](https://arxiv.org/pdf/2606.16180)
- Strategies named in snippets: Retain Label/Max Loss, Noisy Label and Random Encoder. The snippet text was garbled, so the full list of four is unconfirmed. Noisy Label is reported to give the best trade-off: forget-set performance down 93% while retain-set accuracy stays at 84% after 50 epochs. (snippet) — [arXiv pdf](https://arxiv.org/pdf/2606.16180v1)
- Evaluation uses Dice and MAE on the retain and forget sets, plus gap measures between them. No search result mentions membership inference, feature-level auditing or relearning tests. (snippet) — [arXiv 2606.16180](https://arxiv.org/abs/2606.16180)
- No venue or code was found.

**Adjacent work that uses the word "unlearning" in a different sense. None of these is machine unlearning of training data.**
- "Unlearning Scanner Bias for MRI Harmonisation in Medical Image Segmentation" (Oxford). This is domain-adversarial removal of scanner or site information so segmentation generalises across sites. It is not forgetting of training samples. — [ORA Oxford](https://ora.ox.ac.uk/objects/uuid:1ffcbaa8-1f5c-4166-8e86-c3618c895e5f/files/sx920fx23j)
- "Erase to Enhance: Data-Efficient Machine Unlearning in MRI Reconstruction" (OpenReview). The task is reconstruction, not segmentation. It frames unlearning as a way to remove bias. — [OpenReview](https://openreview.net/forum?id=FmCscsj7Ey&noteId=FmCscsj7Ey)

**Non-medical segmentation unlearning, all class-level forgetting.**
- DELETE: "Decoupled Distillation to Erase: A General Unlearning Method for Any Class-centric Tasks", Zhou et al., CVPR 2025, arXiv:2503.23751. It masks forgetting logits apart from retention logits and adds dark-knowledge distillation. It is applied to semantic segmentation, where it reports forgotten-class IoU and retained-class IoU against a retrained model. The segmentation dataset is unverified; snippets mention "dog" and "potted plant" classes, which suggests Pascal VOC. Evaluation is output-only. — [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_Decoupled_Distillation_to_Erase_A_General_Unlearning_Method_for_Any_CVPR_2025_paper.pdf); [arXiv 2503.23751](https://arxiv.org/abs/2503.23751)
- "Class Unlearning via Depth-Aware Removal of Forget-Specific Directions" (DAMP), arXiv:2604.15166, 2026. A search summary says it evaluates class unlearning for segmentation on Pascal VOC 2012 with DeepLabV3 and compares with KDU and SalUn. This is unverified beyond the snippet. Its "forget-specific directions" framing does touch representations. — [arXiv 2604.15166](https://arxiv.org/pdf/2604.15166)

### Inferences
- The medical segmentation unlearning literature amounts to two arXiv preprints, both privacy-motivated (lesions, patients or subjects), both on CNN or U-Net-style models, and both evaluated on outputs only. Neither appears to have passed peer review yet.
- Neither paper uses a corrective, label-noise or poisoned-mask setting. Neither uses a promptable model.

### Gaps
- Full method and evaluation sections of 2606.16180 could not be read because arxiv.org is blocked.
- I found no MICCAI 2024–2026 main-conference paper on unlearning for segmentation. The MICCAI 2025 subject index lists "Maverick: Collaboration-free Federated Unlearning for Medical Privacy", but I could not check whether it covers segmentation. — [MICCAI 2025 categories](https://papers.miccai.org/miccai-2025/categories/)
- I did not search MedIA or IEEE TMI separately. The general searches surfaced nothing from either journal.

## Q2. What exactly is "Erase to Retain"?

### Takeaway
Erase to Retain is an arXiv preprint (2511.16574, 20 Nov 2025) by Nirjhor Datta and Md. Golam Rabiul Alam. It forgets a designated forget subset using adversarial optimization of LoRA adapters inside a teacher-student framework on a U-Net-style network. It is tested on ISIC and CHASE and reports output IoU only. It is not corrective or label-noise unlearning. As far as available sources show, it does not audit internal representations and does not test re-emergence.

### Cited Findings
- Bibliographic details: arXiv:2511.16574, submitted 20 November 2025, cs.CV, by Nirjhor Datta and Md. Golam Rabiul Alam. No peer-reviewed venue was found. — [arXiv 2511.16574](https://arxiv.org/abs/2511.16574v1)
- Method: a frozen backbone with trainable low-rank LoRA adapters in a teacher-student setup. In a "strong unlearning" phase the LoRA modules are adversarially optimised to contradict the teacher's confident predictions on the forget subset. A "gentle restoration" phase follows, with head-only supervised refinement on retained data. — [arXiv html](https://arxiv.org/html/2511.16574v1)
- What is forgotten: the paper describes "efficient and controllable removal of patient specific lesion information". This is a sample or patient-level forget subset motivated by privacy, not corrupted labels. — [arXiv pdf](https://arxiv.org/pdf/2511.16574)
- Results: on ISIC segmentation, forget-set IoU falls from 0.875 to 0.509 while retain and validation IoU stays at 0.647–0.677. On ISIC classification, forget accuracy falls from 87.0% to 64.1% and retain accuracy rises from 83.9% to 90.6%. On cross-domain CHASE (retinal vessels), forget IoU "consistently" falls while utility is preserved. — [arXiv 2511.16574](https://arxiv.org/abs/2511.16574)
- Parameter efficiency: one secondary summary says 0.74M of 25.17M parameters are updated (2.93%). This is unverified against the paper. — [Moonlight review](https://www.themoonlight.io/review/erase-to-retain-low-rank-adaptation-guided-selective-unlearning-in-medical-segmentation-networks)
- Code: github.com/NIRJHOR-DATTA/Erase-to-Retain, as reported in a search summary. I did not open the repository. — [awesomepapers listing](https://awesomepapers.io/computer-vision/papers/2511.16574)
- Evaluation is output-level forget, retain and validation IoU or accuracy. No search result mentions membership inference, feature or logit probing, relearning attacks or prompt robustness. (snippet)

### Inferences
- The objective explicitly pushes forget-set IoU down, to 0.509. In a corrective setting that target would be wrong: the team wants correct segmentation of the polyp on flagged images and removal only of the false ring. Erase to Retain therefore cannot serve as a corrective baseline without changing its objective.
- LoRA-based forgetting keeps the base weights frozen. That makes it a plausible extra candidate for the "suppressed but not removed" residual influence the team wants to audit. This is a hypothesis, not a published finding.

### Gaps
- I could not confirm the backbone (U-Net with which encoder), how large the forget set is, or how it is chosen, because the full text was unreachable.
- I could not confirm whether any appendix includes MIA or internal analysis.

## Q3. Is there unlearning work on SAM, MedSAM, SAM 2, SAM 3 or other promptable segmentation models, including prompt-dependent forgetting?

### Takeaway
Only Du et al. (AAAI 2026, Selective Concept Unlearning, which the team already knows) does machine unlearning on segmentation foundation models. Its target is concept or class removal, not corrective removal of label noise. I found no unlearning paper on MedSAM, SAM 2, SAM 3 or any medical SAM variant. I found no study of prompt-dependent forgetting or re-emergence of forgotten content under prompt perturbation.

### Cited Findings
- "Forget What Has Seen: Selective Concept Unlearning in Segmentation Foundation Models", Miaozeng Du, Jiaqi Li et al., AAAI 2026, DOI 10.1609/aaai.v40i25.39233, pp. 20923–20931. It removes the ability to segment chosen target concepts. Its Multi-level Forgetting Module works at three levels: distillation, attention suppression on target regions, and output relabelling of the target as background. A Preservation Module protects non-target concepts. The paper introduces segmentation-unlearning metrics. The Underline listing says code is promised but not yet released. No arXiv version was found. — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/39233); [Crossref](https://api.crossref.org/works/10.1609%2FAAAI.V40I25.39233); [Underline](https://underline.io/lecture/141429-forget-what-has-seen-selective-concept-unlearning-in-segmentation-foundation-models)
- Du et al. name the main difficulty as entanglement: target features are "tightly entangled with background content and non-target concepts". This is directly relevant to the team's hypothesis that ring influence persists in shared features. — [ResearchGate](https://www.researchgate.net/publication/402628423_Forget_What_Has_Seen_Selective_Concept_Unlearning_in_Segmentation_Foundation_Models)
- Searches combining MedSAM or SAM 2 with unlearning or concept erasure returned only adaptation papers such as Medical SAM 2 (arXiv 2408.00874) and SAMed-2 (MICCAI 2025). None does unlearning. — [Medical SAM 2](https://arxiv.org/abs/2408.00874); [SAMed-2](https://papers.miccai.org/miccai-2025/paper/2056_paper.pdf)
- Work on forgetting in SAM fine-tuning concerns catastrophic forgetting, not deliberate unlearning:
  - CoSAM/MoDA continual learning (arXiv 2412.06418). — [arXiv 2412.06418](https://arxiv.org/html/2412.06418v2)
  - Uncertainty-aware fine-tuning of segmentation foundation models (OpenReview). — [OpenReview](https://openreview.net/forum?id=qNXRXUC90b)
  - "Practical Continual Forgetting for Pre-trained Vision Models" (arXiv 2501.09705). This is deliberate continual forgetting in pre-trained vision models. Whether it covers SAM or segmentation is unverified. — [arXiv 2501.09705](https://arxiv.org/pdf/2501.09705)
- Prompt sensitivity of SAM is well documented outside unlearning:
  - PP-SAM (CVPRW 2024, arXiv 2405.16740) finds zero-shot SAM "highly vulnerable" to bounding-box prompt perturbation on polyp segmentation. — [PP-SAM](https://arxiv.org/html/2405.16740v1)
  - A MICCAI 2026 UNSURE workshop paper studies when prompt-perturbation uncertainty catches interactive-segmentation failures in SAM and MedSAM. — [UNSURE 2026](https://papers.miccai.org/miccai-2026-sat/UNSURE2026_002.html)

### Inferences
- Prompt perturbation as a probe for residual influence after unlearning appears unstudied. The team can cite PP-SAM and the UNSURE 2026 paper to justify perturbation-sensitivity as a known failure axis, and the AAAI SCU paper as the only SAM unlearning precedent.
- Du et al. erase a whole concept by relabelling it as background. The team instead removes a spurious structure (the ring) attached to a correct concept (the polyp). Neither SCU's task definition nor its metrics, as described in the abstract, cover that case.

### Gaps
- I could not confirm which SAM variants SCU uses (SAM ViT-B/H, SAM 2, HQ-SAM and so on), which datasets, or whether it probes features or tests relearning. The full text is unavailable and ResearchGate has no full text. The team should read the AAAI PDF.
- I found nothing on SAM 3 unlearning. This may mean none exists, or search coverage may be incomplete.

## Q4. Is there corrective unlearning, or unlearning of noisy or poisoned labels, in segmentation or medical imaging, including backdoor or poison removal in segmentation?

### Takeaway
I found no paper that applies corrective unlearning (Goel et al.'s setting, with partial identification of corrupted data) to segmentation or to medical imaging. Segmentation label noise is handled by robust training and label correction (ADELE and its successors), not by post-hoc unlearning. Segmentation backdoor repair exists, in Segment Cleanse and RePAIR, but it targets trigger-based backdoors on non-medical models. Evidence from classification shows that unlearning often fails to remove poisons, which supports the premise of the team's audit.

### Cited Findings
- Noisy-label medical segmentation is treated by training-time correction, not unlearning:
  - Adaptive Label Correction (arXiv 2503.12218). — [arXiv 2503.12218](https://arxiv.org/abs/2503.12218)
  - Clean Label Disentangling (arXiv 2311.16580). — [arXiv 2311.16580](https://arxiv.org/pdf/2311.16580)
  - MTCL (Mean-Teacher-assisted Confident Learning). — [PubMed 35604969](https://pubmed.ncbi.nlm.nih.gov/35604969)
  - Confident-learning label correction for retinal segmentation, Diagnostics 2025. — [DOI 10.3390/diagnostics15141735](https://doi.org/10.3390/diagnostics15141735)
  - Co-Seg (arXiv 2102.00523). — [arXiv 2102.00523](https://arxiv.org/pdf/2102.00523)
- RePAIR: "Inverting Backdoor Attacks in Semantic Segmentation" (arXiv 2610.37015, Oct 2026). A post-hoc, label-free repair of a frozen segmentation model against targeted segmentation backdoors, using an unlabeled image pool that may contain triggered samples. It notes that the reference data used for repair must itself be protected from poisoned samples. — [arXiv 2610.37015](https://arxiv.org/html/2610.37015)
- Segment Cleanse (Mao et al., 2023, Computers & Security) adapts Neural Cleanse to semantic segmentation through object-free backdoor attack and defense. It reconstructs trigger candidates and mitigates the backdoor by retraining. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167404823002754)
- "Machine Unlearning Fails to Remove Data Poisoning Attacks", Pawelczyk et al., ICLR 2025, arXiv 2406.17216. Across image classifiers and LLMs, existing unlearning methods fail to remove indiscriminate, targeted and Gaussian poisons even with large compute. GD-based unlearning helps only against indiscriminate poisoning in vision. The authors introduce poison-based unlearning evaluation metrics and warn of a "false sense of confidence". — [arXiv 2406.17216](https://arxiv.org/abs/2406.17216v3); [ICLR 2025](https://iclr.cc/virtual/2025/poster/30218)
- Classification backdoor-unlearning methods: Burn (arXiv 2507.10491), FLARE (arXiv 2411.19479) and "Backdoor Unlearning by Linear Task Decomposition" (arXiv 2510.14845). — [Burn](https://arxiv.org/html/2507.10491); [FLARE](https://arxiv.org/pdf/2411.19479); [Linear Task Decomposition](https://arxiv.org/html/2510.14845)
- DELETE (CVPR 2025) also lists backdoor defense as an application, but in class-centric form. — [arXiv 2503.23751](https://arxiv.org/abs/2503.23751)

### Inferences
- The team's setting combines a synthetic, spatially structured mask corruption (the false ring), partial flagging and post-hoc unlearning. That combination is equivalent to a clean-trigger-free, label-side poison in segmentation. It sits between Goel et al.'s corrective unlearning and segmentation backdoor repair, and neither literature covers it.
- Pawelczyk et al. supply the strongest prior motivation for the team's question. RePAIR and Segment Cleanse are the nearest segmentation analogues, but they assume input triggers and do not use promptable models.

### Gaps
- I did not search specifically for "spurious structure" or "shortcut" unlearning in segmentation, such as removing a reliance on rulers in dermoscopy. That could be a close neighbour.
- RePAIR's evaluation, and whether it audits internal features, could not be read.

## Q5. What practice does medical-imaging unlearning more broadly, mainly classification, set for evaluation?

### Takeaway
Medical unlearning papers outside segmentation use patient-level forget sets and evaluate with forget and retain task metrics, membership inference attacks (MIA) and comparison to retraining. In parallel, general vision unlearning work from 2025–2026 shows that residual knowledge survives in features and returns under relearning or perturbation. Nobody has yet carried those residual-knowledge audits into medical segmentation.

### Cited Findings

**Medical unlearning papers.**
- Forget-MI (Hardan, Taratynova, Essofi, Nandakumar, Yaqub; MBZUAI; MICCAI 2025; arXiv 2506.23145). Patient-level forget sets of 3%, 6% and 10% of the data on multimodal image and text records. It reports MIA reduced by 0.202, forget-set AUC down 0.221 and F1 down 0.305, with test performance comparable to retraining. Code is at BioMedIA-MBZUAI/Forget-MI. — [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/3777_paper.pdf); [arXiv 2506.23145](https://arxiv.org/abs/2506.23145v1)
- "Machine Unlearning for Medical Imaging" (Nasirigerdeh, Razmi, Schnabel, Rueckert, Kaissis; arXiv 2407.07539, Jul 2024; no venue found). Benchmarks the effectiveness and efficiency of existing algorithms on medical classification. It finds the approach promising but in need of better generalisation and hyperparameter tuning, and reports possible bias with respect to easy or hard samples. — [arXiv 2407.07539](https://arxiv.org/abs/2407.07539?context=cs)
- GRIN+ (Minghui Huang, Junxiao Wang; arXiv 2609.15571, 2026). Unlearning for imbalanced medical data, framed as a privacy-efficiency-utility trilemma, with parameter-level separation of unlearning-specific knowledge. Its datasets and metrics are unverified. — [arXiv 2609.15571](https://arxiv.org/pdf/2609.15571)
- Maverick: Collaboration-free Federated Unlearning for Medical Privacy (MICCAI 2025). Listed in the subject index; content not retrieved. — [MICCAI 2025 categories](https://papers.miccai.org/miccai-2025/categories/)
- "Towards Reliable Forgetting: A Survey on Machine Unlearning Verification" (arXiv 2506.15115). Useful framing for verification methods. — [arXiv 2506.15115](https://arxiv.org/pdf/2506.15115)

**Residual-knowledge and relearning audits in general vision, the methodological precedent for the team's audit.**
- "Unlearning's Blind Spots: Over-Unlearning and Prototypical Relearning Attack" (arXiv 2506.01318). Class-unlearned image classifiers keep clustered forget-class features, and a few samples are enough to relearn them. — [arXiv 2506.01318](https://arxiv.org/pdf/2506.01318)
- RUB: "Evaluating Residual Knowledge in Unlearned Models" (arXiv 2504.14798). Introduces the Unlearning Mapping Attack and covers classification, image-to-image and text-to-image models. — [arXiv 2504.14798](https://arxiv.org/html/2504.14798)
- "From Dormant to Deleted: Tamper-Resistant Unlearning Through Weight-Space Regularization" (arXiv 2505.22310). Relearning resistance is predicted by L2 distance and linear mode connectivity to the original model. Output-level and representation-level unlearning methods are less robust. — [arXiv 2505.22310](https://arxiv.org/html/2505.22310v1)
- "The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples" (RURK). Defines residual knowledge using perturbed forget samples relative to a retrained model and finds that most methods exceed the retrain baseline. — [Lacuna summary](https://lacuna.tiptreesystems.com/work/the-unseen-threat-residual-knowledge-in-machine-unlearning-under-perturbed/wrk_4dc422955a9ab7bf43b7c3b7436bd200)
- Reminiscence Attack (arXiv 2507.20573). Approximate unlearning leaves latent residuals in the loss landscape even when outputs match retraining. — [Pith](https://pith.science/paper/2507.20573)
- "Memory Self-Regeneration: Uncovering Hidden Knowledge in Unlearned Models" (arXiv 2510.03263). — [arXiv 2510.03263](https://arxiv.org/html/2510.03263)
- "A Durable Machine Unlearning Framework to Nullify Recall of Sensitive Data on Incremental Training" (IJCAI 2026). Addresses re-emergence under continued training. — [IJCAI 2026](https://ijcai.org/proceedings/2026/715)

### Inferences
- A strong evaluation design for the team would combine three things:
  1. Standard output metrics against retrain-on-clean and retrain-minus-flagged, following Goel et al. and Forget-MI practice.
  2. Pixel-localised internal probes on the ring region: pre-threshold logits and decoder or encoder features. This is the spatial counterpart to the feature-cluster residuals in Blind Spots and RUB.
  3. Relearning and perturbation tests: continued fine-tuning following Dormant-to-Deleted and Durable Unlearning, and prompt perturbation in the spirit of RURK's perturbed samples.
- None of the medical papers found uses parts 2 or 3.

### Gaps
- Whether Forget-MI or Nasirigerdeh et al. include any representation-level analysis is unknown, because I read only summaries.

## Conclusion: which elements of the team's study are done, partly done, or not found

| Study element | Status | Closest prior work |
|---|---|---|
| Unlearning in medical image segmentation | **Partly done.** Two preprints exist, both privacy-motivated, on CNNs or U-Net, with output-only evaluation. | Erase to Retain (2511.16574); To forget is to preserve (2606.16180) |
| Unlearning in SAM-family segmentation foundation models | **Partly done.** One paper, focused on concept removal. | Du et al., AAAI 2026 (SCU) |
| Unlearning in SAM fine-tuned on medical data (Kvasir-SEG, MedSAM-like) | **Not found** | None |
| Corrective unlearning (partially flagged corrupted labels) in segmentation | **Not found** | Goel et al. CMU (classification); Pawelczyk et al. ICLR 2025 (classification and LLM); ADELE (training-time noise robustness, not unlearning) |
| Removing a spatially structured spurious mask artefact (the false ring) rather than a class, concept or sample | **Not found.** Segmentation backdoor repair is adjacent, but it uses input triggers. | RePAIR (2610.37015); Segment Cleanse (2023) |
| SSD and SAP baselines applied to segmentation | **Not found** for segmentation. SAP (Kodge et al.) and SSD exist only for classification in the sources found. | — |
| Internal or residual-influence audit (logits, features) after unlearning | **Done in classification and generative models.** **Not found** for segmentation. **Not found** in pixel-localised or spatial form. | Blind Spots (2506.01318); RUB (2504.14798); Reminiscence (2507.20573) |
| Re-emergence under continued fine-tuning | **Done in classification and LLMs.** **Not found** in segmentation. | Dormant-to-Deleted (2505.22310); IJCAI 2026 Durable Unlearning; Blind Spots |
| Re-emergence or residual probing under prompt perturbation | **Not found.** Prompt fragility of SAM itself is well documented. | PP-SAM (CVPRW 2024); UNSURE 2026 SAM/MedSAM prompt-perturbation study |

Overall: no paper found combines corrective, partially flagged unlearning with a promptable segmentation model and a spatially localised internal audit. The main risks to the novelty claim are two things that could not be read in full:
1. Du et al.'s SCU paper may contain attention- or feature-level analyses, given that it has an attention-suppression module.
2. A late-2026 preprint may have been missed, because search coverage of arXiv from October to November 2026 is uncertain.

The team should read the SCU full text and run one more arXiv listing search for 2026 before finalising the claim.
