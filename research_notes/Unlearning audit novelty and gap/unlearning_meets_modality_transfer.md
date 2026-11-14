# Unlearning meets cross-modality transfer and prompt-type evaluation in promptable medical segmentation

Scope note: searched up to 14 Nov 2026. In this session WebFetch could not reach arxiv.org, huggingface.co, emergentmind.com, ojs.aaai.org, web3.arxiv.org or zenodo.org (DNS blocked by the sandbox). github.com could be fetched. So many claims below come from search-engine summaries of primary pages, not from reading full texts. Those claims are marked "(search summary)". Before anything goes into a proposal, check them against the PDFs. Section 4.1 of Medical SAM3, which has the dataset tables, is the most important one to check.

## Q1. Is there published work on unlearning or forgetting a modality, dataset or domain in medical segmentation foundation models? Does adapting to ultrasound or dermoscopy cause forgetting?

### Takeaway
Two bodies of work exist, but nobody has joined them. Continual learning for SAM-family medical models treats cross-modality catastrophic forgetting as a known problem (SAMed-2, EvoSAM, CA-SAM). Machine unlearning for medical segmentation exists only for non-promptable networks (subject-level unlearning on brain MRI; LoRA "Erase to Retain" on ISIC 2018). I found no paper on deliberate modality-level or dataset-level unlearning in a promptable medical foundation model such as MedSAM, Medical SAM3 or MedSAM3. I also found no direct measurement of how much fine-tuning on ultrasound or dermoscopy degrades other modalities.

### Cited Findings
- SAMed-2 (MICCAI 2025, arXiv 2507.03698) builds on SAM-2 and adds a confidence-driven memory that keeps high-certainty features to limit forgetting as new tasks or modalities arrive. It was trained and evaluated on MedBank-100k (7 modalities, 21 tasks). It frames the problem as "continuous learning across different organs and modalities introduces the risk of catastrophic forgetting" and reports a 10.53% gain in external zero-shot results (search summary) — [SAMed-2](https://arxiv.org/pdf/2507.03698)
- EvoSAM (arXiv 2503.06236) is a continual-learning method for SAM/MedSAM. It argues that CNN-era continual-learning methods (e.g., lifelong nnU-Net) can overfit or catastrophically forget when applied to SAM. It compares against sequential fine-tuning and experience replay, and its experiments cover surgical blood-vessel segmentation and multi-centre prostate MRI. These are not ultrasound or dermoscopy (search summary) — [EvoSAM](https://arxiv.org/pdf/2503.06236)
- CA-SAM (arXiv 2511.17201, "Continual Alignment for SAM") is exemplar-free. It uses learnable alignment layers with VAE-based task routing and falls back to frozen SAM on unseen data. It was evaluated on 9 medical segmentation datasets (search summary) — [CA-SAM](https://arxiv.org/html/2511.17201v1)
- SUM (NSF PAR copy) states that fine-tuning SAM can erode its general segmentation ability on tasks outside the fine-tuning set, and it adds task prompts to reduce this (search summary) — [SUM, par.nsf.gov](https://par.nsf.gov//servlets/purl/10649437)
- Subject-level machine unlearning for 3D segmentation (arXiv 2606.16180, June 2026) defines the forget set as all samples from designated subjects. It requires poor forget-set performance while keeping retain-set accuracy close to the original model and to a model retrained from scratch. It benchmarks approximate unlearning methods on MRBrainS18 brain MRI. This is not a foundation model and has no prompts (search summary) — [To forget is to preserve](https://arxiv.org/html/2606.16180)
- "Erase to Retain" (arXiv 2511.16574) is LoRA-guided selective unlearning for medical segmentation networks. It freezes a teacher, trains only low-rank adapters (0.74M of 25.17M parameters, 2.93%), and pushes forget-set predictions toward pure background. It is evaluated on ISIC 2018 (dermoscopy) and CHASE_DB1 (retinal vessels), and it claims to be the first LoRA-based unlearning framework for dense medical segmentation (search summary) — [Erase to Retain](https://arxiv.org/html/2511.16574v1)
- MedSAM's model is trained on about 1.57M image-mask pairs across 10 modalities — [MedSAM, Nat Commun 2024 (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10803759/)
- In ultrasound comparisons, MedSAM "performs the worst" despite large-scale medical pretraining, and UltraSam notes that MedSAM, SAM-Med and Medical SAM Adapter "continue to struggle with ultrasound" (search summary). This is evidence of a modality transfer gap, not of forgetting — [UltraSam, arXiv 2411.16222](https://arxiv.org/html/2411.16222v1)
- UniUltra (arXiv 2511.15771) is an ultrasound-specific parameter-efficient SAM2 — [UniUltra](https://arxiv.org/pdf/2511.15771)

### Inferences
- "Forgetting a modality or dataset in a promptable medical foundation model" looks unpublished as an unlearning problem. The nearest work uses continual learning, where forgetting is something to avoid, not a goal.
- Erase to Retain overlaps directly with dermoscopy unlearning (ISIC 2018). Any dermoscopy unlearning study must cite it and set itself apart from it: Erase to Retain is not promptable, has no prompt-type axis, and does forget-to-background rather than corrective unlearning.
- Cheap, defensible experiment: fine-tune Medical SAM3 on a new ultrasound set, then measure the Dice drop on its other modalities (including Kvasir/CVC endoscopy). That gives a forgetting matrix, which doubles as a "transfer cost" result for the assignment card.

### Gaps
- I found no paper that reports per-modality forgetting after ultrasound-only or dermoscopy-only fine-tuning of SAM, MedSAM or SAM3.
- I could not read the EvoSAM or CA-SAM tables to see whether they include ultrasound or dermoscopy tasks.

## Q2. Does unlearning or corrupted-label correction behave differently per modality or per prompt type (point, box, text)? Does prompt type change how much forgotten behaviour comes back?

### Takeaway
I found no empirical study of prompt type as a variable in unlearning efficacy for SAM-style models. The general unlearning literature shows that forgotten knowledge often survives and comes back under perturbed inputs ("residual knowledge") or after a few relearning steps. Prompt type is a natural "input perturbation" for promptable models, so this is an open and testable gap.

### Cited Findings
- "The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples" (NeurIPS 2025, arXiv 2601.22359) defines residual knowledge as traces of forgotten data that persist and reappear under perturbed samples. It reports that this is prevalent across existing unlearning methods and proposes a fine-tuning fix, RURK (search summary) — [NeurIPS 2025 proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/2656bba937d78593fbd99ace9f14e311-Abstract-Conference.html); [arXiv 2601.22359](https://arxiv.org/pdf/2601.22359)
- A Prototypical Relearning Attack restores forgotten-class performance from a few forget-class samples. The paper proposes the defence "Spotter" against relearning and over-unlearning (search summary) — [arXiv 2506.01318](https://arxiv.org/html/2506.01318)
- SCU (AAAI 2026) is reported as "the first to study machine unlearning in segmentation foundation models" (search summary). I could not verify which prompt types it evaluates — [AAAI OJS 39233](https://ojs.aaai.org/index.php/AAAI/article/view/39233)
- Medical SAM3 reports that vanilla SAM3 does poorly on medical data and "leans heavily on geometric priors such as ground-truth bounding boxes" (search summary). This means box prompts carry much of the localisation signal — [Medical SAM3 HTML](https://arxiv.org/html/2601.10880)
- MedSAM3 (arXiv 2511.19046, a different model from Medical SAM3) reports that text-only prompting was insufficient and evaluates text + geometry (box) prompts on BUSI, RIM-ONE, ISIC 2018 and Kvasir-SEG (search summary) — [MedSAM3](https://arxiv.org/html/2511.19046v2)
- A few-shot concept prompt-learning paper (alphaxiv 2608.01663) reports that text prompts are brittle on ultrasound, and that adapting Medical SAM3 with its method beats frozen baselines on TN3K, CVC-Clinic and HC18 (search summary) — [Few-Shot Concept Prompt Learning](https://www.alphaxiv.org/abs/2608.01663)

### Inferences
- Hypothesis worth testing: box prompts may "override" unlearning. A box supplies strong geometric evidence, so a model whose text-conditioned behaviour was unlearned or corrected may still reproduce the old masks when given a box, and possibly a point. That would be a prompt-conditional form of residual knowledge. Text prompts route through the concept or presence pathway in SAM 3, so they may show the cleanest forgetting.
- Modality could moderate this. Text is reported to be brittle on ultrasound, so text-prompt unlearning effects may be confounded by low baseline performance there. Report the unlearning metric relative to the baseline for each (modality, prompt) cell.
- The team's existing design (`design/prompt_conditions.csv`, `recovery_finetune.csv`) already appears to vary prompt conditions and measure relearning. That maps directly onto this gap, so the main change is moving from SAM ViT-B on Kvasir to Medical SAM3 with text prompts.

### Gaps
- I found no paper that measures unlearning efficacy separately for point, box and text prompts.
- I found no paper that compares unlearning or corrective-unlearning efficacy across imaging modalities.

## Q3. Is text-prompt (concept) unlearning studied for SAM 3-style promptable concept segmentation?

### Takeaway
Concept unlearning has been published for segmentation foundation models: SCU at AAAI 2026, apparently on SAM-type models with natural-image classes such as "fighter jet". I found nothing on unlearning in SAM 3 itself, which uses noun-phrase concept prompts, and nothing in medical imaging. SAM 3 work on medical images so far covers adaptation, prompt drift and test-time adaptation, not forgetting.

### Cited Findings
- SCU, "Forget What Has Seen: Selective Concept Unlearning in Segmentation Foundation Models", by Miaozeng Du, Jiaqi Li et al., AAAI-26, Proc. AAAI 40(25):20923–20931, DOI 10.1609/AAAI.V40I25.39233. It removes the ability to segment target concepts while a Preservation Module keeps non-target concepts working. Its example is forgetting "fighter jet" while still segmenting bicycles. It works at three levels, one of which relabels target predictions as background. It also introduces unlearning metrics for segmentation and says code will be released "in the near future" (search summary; I could not open the full text) — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/39233); [Crossref](https://api.crossref.org/works/10.1609%2FAAAI.V40I25.39233); [Underline talk](https://underline.io/lecture/141429-forget-what-has-seen-selective-concept-unlearning-in-segmentation-foundation-models)
- SAM 3, "Segment Anything with Concepts" (arXiv 2511.16719; ICLR 2026), uses concept prompts that are short noun phrases, image exemplars, or both. It has a detector and tracker that share a backbone, plus a presence head that separates recognition from localisation (search summary) — [SAM 3 arXiv](https://arxiv.org/pdf/2511.16719); [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html)
- ConceptBank (arXiv 2602.06333) shows SAM 3 failing when a static text prompt no longer matches visual content in new domains ("data drift" and "concept drift"). It restores prompt-mask alignment by replacing source-induced prompt embeddings. This is adaptation, not unlearning (search summary) — [arXiv 2602.06333](https://arxiv.org/pdf/2602.06333)
- Test-time adaptation of SAM3 for medical segmentation (arXiv 2606.22963) uses text prompts such as "the skin lesion" for ISIC2018 and "the prostate" for Promise12 (search summary) — [arXiv 2606.22963](https://arxiv.org/html/2606.22963)
- Other SAM 3 medical works found: "Adapting SAM 3 for concept-driven lesion segmentation" (PMC13658689; uses Kvasir-SEG, ISIC2018, BUSI), "From SAM 1 to SAM 3: Benchmarking zero-shot cross-domain medical image segmentation" (Expert Systems 2026), and a SAM 3 LoRA repo on MedSegBench — [PMC13658689](https://pmc.ncbi.nlm.nih.gov/articles/PMC13658689); [Expert Systems](https://onlinelibrary.wiley.com/doi/10.1111/exsy.70383); [GitHub sam3-lora](https://github.com/Good-SAMaritans/sam3-lora-medical-segmentation)
- ConceptSeg-R1 (arXiv 2605.20385) does concept segmentation through meta-RL. It is not an unlearning paper — [arXiv 2605.20385](https://arxiv.org/pdf/2605.20385)

### Inferences
- "Text-concept unlearning in Medical SAM3", for example forgetting the concept "skin lesion" or "breast tumor" or one dataset's label vocabulary, is not covered by SCU because SCU is not medical and does not use SAM 3 noun-phrase prompts. It would be a novel extension. However, SCU is the closest prior art and must be the main baseline or comparison.
- SAM 3's presence head separates "is the concept here?" from "where is it?". A concept-unlearning attack surface is therefore: text unlearning suppresses presence, but a box or exemplar prompt bypasses presence and recovers the mask. This is a specific, mechanistic hypothesis for Q2.

### Gaps
- I could not confirm SCU's backbone (SAM, SAM 2 or SAM 3), its prompt types, its datasets, or whether code is released.
- I found no SAM 3 unlearning paper as of November 2026.

## Q4. Annotation-error characteristics in ultrasound (BUSI etc.) and dermoscopy (ISIC, PH2, HAM10000)

### Takeaway
Both modalities have documented, real annotation defects, so corrective unlearning can use natural forget sets instead of synthetic corruption. BUSI has duplicated images with conflicting masks, non-breast scans, and inconsistencies; a curated subset keeps 450 of 780 images. ISIC masks were made with three different tools that leave distinct boundary signatures. The new IMA++ multi-annotator release records each mask's tool and annotator skill level.

### Cited Findings
- A Data in Brief letter (2023) on BUSI reports duplicated lesion images whose reference masks disagree, recommends removing duplicates, and notes that the set includes axilla scans not clearly described. The letter's figure of "235 duplicated cases" may refer to a broader collection, so check it before quoting — [Letter to the Editor, PMC10293973](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10293973/)
- Curated BUSI (Aumente-Maestro et al., 2024) keeps 450 of 780 images after removing duplicates and inconsistent annotations (search summary, via a repository index) — [multi_task_breast_cancer index](https://awesome.ecosyste.ms/projects/github.com%2Fcaumente%2Fmulti_task_breast_cancer); related curation paper [arXiv 2407.11316](https://arxiv.org/pdf/2407.11316v2.pdf)
- arXiv 2508.17768 finds that BUSI duplicates cause train/test leakage. Mean Dice falls from 0.7514 on full BUSI to about 0.71–0.72 on three deduplicated variants, one of which keeps the copy a radiologist judged most accurate. It also mentions maxilla/axilla images (search summary) — [arXiv 2508.17768](https://arxiv.org/html/2508.17768v1)
- BrEaST is a curated breast ultrasound alternative: 256 scans from 256 patients, each manually annotated by a radiologist (search summary) — [BrEaST summary](https://bohrium.dp.tech/paper/arxiv/246e4fcd8bfa4afda44d49ac13058883f820c2d0fecc1cbab36cdf43e5f318a2)
- IMA++ (arXiv 2512.21472) is a multi-annotator dermoscopic segmentation dataset from the ISIC Archive. Each mask is tagged with its creation method (manual polygon, semi-automated flood-fill, or automated with expert review) and with reviewer skill (expert or novice). Manual polygon and automated masks make up about 81% of annotations (search summary) — [IMA++](https://arxiv.org/pdf/2512.21472)
- Abhishek et al., "What can we learn from inter-annotator variability in skin lesion segmentation?" (arXiv 2508.09381), report a significant link (p<0.001) between inter-annotator Dice agreement and lesion malignancy, and analyse variability by annotator, tool and skill (search summary) — [arXiv 2508.09381](https://arxiv.org/html/2508.09381v1)
- Earlier work observes that flood-fill masks have very irregular borders, manual polygons are very smooth, and automated-plus-expert masks fall in between with pixelated borders. An ISIC 2017 study found some ground-truth masks were inaccurate, especially automatically generated ones (search summary) — [Less is More, arXiv 2004.13856](https://arxiv.org/pdf/2004.13856); [Ribeiro et al., arXiv 1906.02415](https://arxiv.org/pdf/1906.02415)

### Inferences
- Natural forget sets are available:
  - BUSI: the conflicting-mask duplicates and the images removed in curation. The 330 removed images, compared against the curated 450, form a ready-made "erroneous subset".
  - ISIC/IMA++: flood-fill or novice-reviewed masks as "suspect" labels, with expert masks of the same image as the corrected target.
- Using these turns the synthetic-corruption audit into a corrective-unlearning-with-real-errors audit. It is more clinically credible and it naturally covers ultrasound and dermoscopy.
- Caveat: since BUSI or ISIC may be in Medical SAM3's training data (see Q5), corrective unlearning on them is actually meaningful there. The model has really learned from those noisy labels. In a Kvasir/SAM ViT-B setup with synthetic noise, the noise must first be injected by fine-tuning.

### Gaps
- I did not verify inter-rater statistics for BUSI or for PH2. Whether PH2 has a single expert annotator is unverified.
- I could not confirm which ISIC challenge years (2016, 2017, 2018) share images with IMA++, which matters for leakage.

## Q5. Dataset availability and licensing; did Medical SAM3 train on these datasets (leakage risk for "untested modality")?

### Takeaway
This is the most consequential finding. Ultrasound and dermoscopy are not untested modalities for Medical SAM3. Search summaries of the paper say it was fine-tuned on 33 datasets across 10 modalities that include ultrasound and dermoscopy. Its zero-shot external set includes TN3K (thyroid ultrasound), HC18 (fetal ultrasound) and PH2 (dermoscopy). So the card's "transfer to an untested public modality (ultrasound, dermoscopy)" premise does not hold literally for this model. The team must either pick truly unseen datasets, or reframe the work as "unseen dataset within a seen modality". The repo releases code and weights and supports text and box prompts; point prompts are not mentioned.

### Cited Findings
- Medical SAM3 is arXiv:2601.10880, "Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation", by Chongcong Jiang, Yu Tian et al., submitted 15 Jan 2026. It was fine-tuned on 33 datasets across 10 modalities, reported as including radiography, ultrasound, endoscopy, pathology, fundus and dermoscopy (search summary of the paper and of chatpaper) — [arXiv abs](https://arxiv.org/abs/2601.10880); [chatpaper](https://chatpaper.com/paper/227169); [papers.cool](https://papers.cool/arxiv/2601.10880)
- The external zero-shot datasets, excluded from training, are reported as TN3K, HC18, CHASE_DB1, STARE, CVC-Clinic, ETIS-Larib and PH2. There are 10 internal held-out tasks, and a BUSI row appears in the dataset table, but the snippet did not show whether it is internal or training (search summary of the arXiv HTML) — [Medical SAM3 HTML](https://arxiv.org/html/2601.10880v1)
- Reported external averages are 73.9% Dice and 64.4% IoU, versus 11.9% Dice and 8.0% IoU for vanilla SAM3. The reported internal held-out mean is 77.0% Dice, versus 54.0% for SAM3 with text-only prompts. Training reportedly uses only text and masks, with no spatial prompts. These numbers come from aggregators (Liner, Emergent Mind), not the PDF — [Liner review](https://liner.com/review/medical-sam3-foundation-model-for-universal-promptdriven-medical-image-segmentation); [Emergent Mind topic](https://api.emergentmind.com/topics/medical-sam3)
- The GitHub repo (AIM-Research-Lab/Medical-SAM3) releases 2D and 3D inference, training and evaluation code. The news log lists pretrained weights on 2026-01-20, with a 3D checkpoint on Hugging Face. Text prompts work in 2D and 3D (e.g., `--prompt '1=liver'`), and 2D also takes box prompts. Point prompts are not mentioned. The README names CHASE_DB1, STARE and CVC-ClinicDB as 2D benchmarks and does not list BUSI, ISIC, PH2 or TN3K. A LICENSE file exists, but its type was not shown in the fetched page (fetched directly) — [GitHub Medical-SAM3](https://github.com/AIM-Research-Lab/Medical-SAM3)
- MedSAM3 (arXiv 2511.19046) is a different model; do not confuse the two. It fine-tunes on BUSI, RIM-ONE, ISIC 2018 and Kvasir-SEG, using text prompts such as "breast tumor". Reported numbers: SAM 3 with text + geometry gets 0.7110 Dice on BUSI and 0.8178 on ISIC 2018; fine-tuned MedSAM-3 gets 0.7772 on BUSI. The summaries contain internally inconsistent numbers, so treat them with caution (search summary) — [MedSAM3](https://arxiv.org/html/2511.19046v2)
- BUS-BRA's Zenodo record (v1.0, 13 Mar 2023) says PEB/COPPE-UFRJ (Brazil) holds the copyright and distributes the data, and that research using it must cite the dataset paper. A specific CC licence was not visible in the snippet (search summary; Zenodo fetch blocked) — [Zenodo 7730708](https://zenodo.org/record/7730708)

### Inferences
- Leakage risks:
  - BUSI: probably in Medical SAM3 training or internal data.
  - ISIC: dermoscopy is a training modality, so some ISIC year is likely used.
  - TN3K, HC18, PH2: reported as external, so held out from training.
  - BUS-BRA, BrEaST, DDTI and IMA++: not mentioned anywhere I found, so they are candidates for truly unseen datasets. This must be verified against Section 4.1.
- Recommended reframing for the card:
  - "Untested modality" becomes either a truly unseen modality, or unseen datasets in seen modalities with a documented leakage check. Truly unseen modality options: none of the 10 listed is certain. Possibly OCT or thermal imaging, but this is unverified.
  - The leakage check should cover duplicate or near-duplicate images, since BUSI is known to have internal duplicates and BUS-BRA/BUSI may share sources (unverified).
- Prompt-type axis: point prompts are not documented in the Medical SAM3 repo. "Point" may require falling back to SAM3's native interactive interface or to the original SAM. Medical SAM3 was trained with text only, so box and point behaviour is effectively out-of-distribution for its fine-tuning. That itself is a finding worth reporting.

### Gaps
- I could not read Medical SAM3's dataset table directly, so it is unverified which ultrasound and dermoscopy datasets (BUSI, ISIC 2016/2017/2018, HAM10000) are in training.
- Licences are unverified in this session. From prior knowledge, not checked: BUSI is published in Data in Brief (Elsevier open access); ISIC 2018 and HAM10000 are commonly distributed under CC BY-NC 4.0; PH2 requires a request form for research use; TN3K comes from Gong et al. (ISBI 2021) via GitHub; DDTI is openly downloadable. All of these need confirmation from the dataset pages.
- The type of the Medical SAM3 repo's LICENSE was not visible.

## Q6. Candidate hybrid research questions, with closest prior art and novelty

### Takeaway
The strongest fusion keeps the card's reproduction, transfer and prompt-type skeleton and adds unlearning as the evaluation lens. The lead question is RQ1: does corrective unlearning in Medical SAM3 hold up across prompt types and modalities? It has the clearest gap, re-uses the team's existing prompt-condition and relearning design, and delivers the card's "per-modality prompt-selection protocol and failure taxonomy" with an unlearning dimension added.

### Cited Findings
- Prior-art anchors, all cited above:
  - SCU, concept unlearning in segmentation foundation models — [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39233)
  - Erase to Retain, LoRA unlearning on ISIC 2018 — [arXiv 2511.16574](https://arxiv.org/html/2511.16574v1)
  - Subject-level 3D unlearning — [arXiv 2606.16180](https://arxiv.org/html/2606.16180)
  - Residual knowledge under perturbations — [NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/2656bba937d78593fbd99ace9f14e311-Abstract-Conference.html)
  - Continual learning for SAM — [SAMed-2](https://arxiv.org/pdf/2507.03698), [EvoSAM](https://arxiv.org/pdf/2503.06236), [CA-SAM](https://arxiv.org/html/2511.17201v1)
  - BUSI curation and duplicates — [PMC10293973](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10293973/), [arXiv 2508.17768](https://arxiv.org/html/2508.17768v1)
  - IMA++ multi-annotator ISIC — [arXiv 2512.21472](https://arxiv.org/pdf/2512.21472)

### Inferences (proposed RQs)

**RQ1 (recommended): Prompt-conditional residual knowledge.** After corrective unlearning of erroneous masks in Medical SAM3, does the corrected behaviour hold under text, box and point prompts, and on ultrasound vs dermoscopy vs the base benchmark?
- Design:
  1. Reproduce Medical SAM3 text-prompt results on its public benchmark.
  2. Build forget sets for each modality. Synthetic corruption on Kvasir/CVC as the control; real errors on BUSI (conflicting duplicates or curated-out images) and ISIC/IMA++ (flood-fill or novice masks).
  3. Apply 2–3 unlearning methods (e.g., gradient ascent or relabel-to-reference, a LoRA-based method in the style of Erase to Retain, SCU-style preservation).
  4. Evaluate a modality × prompt-type grid for forget efficacy, retain utility, and recovery after a few relearning steps.
- Closest prior art: SCU (natural images, concept forgetting, prompt types unverified); Erase to Retain (ISIC, non-promptable); residual knowledge (NeurIPS 2025, classification).
- Novelty: high. I found no work measuring unlearning efficacy as a function of prompt type, or across medical modalities.
- Mechanistic hook: SAM 3's presence head versus geometric prompts that bypass it.
- Card fit: covers reproduction, both modalities, all three prompt types, a prompt-selection protocol ("which prompt type is safest or most reliable after correction, per modality"), and a failure taxonomy (residual leakage, over-forgetting, boundary regression).

**RQ2: Text-concept unlearning in a medical concept-promptable model.** Can Medical SAM3 forget one clinical concept or modality vocabulary (e.g., "skin lesion" or "breast tumor") via text-prompt unlearning without hurting other concepts, and does the forgotten concept come back under box or point prompts or under synonym text prompts?
- Closest prior art: SCU (AAAI 2026); ConceptBank prompt drift (arXiv 2602.06333).
- Novelty: medium-high. It is the first medical and SAM 3 version of SCU, but conceptually close to SCU, so reviewers will see it as "SCU in medicine" unless the prompt-bypass and synonym-leakage analysis is central.
- Clinical motivation is weaker than RQ1 (why forget a lesion type?). Possible framing: data-withdrawal requests, or a site revoking consent for one dataset.

**RQ3: Dataset- or modality-scoped unlearning versus transfer forgetting.** Unlearn one training dataset's contribution (e.g., BUSI, modelling consent withdrawal) from Medical SAM3. Then measure collateral damage to the same modality's other datasets (TN3K, HC18, BUS-BRA) and to other modalities, and compare with the unintentional forgetting caused by fine-tuning on a new ultrasound or dermoscopy set.
- Closest prior art: subject-level 3D unlearning (arXiv 2606.16180, non-foundation); continual learning for SAM (SAMed-2, EvoSAM, CA-SAM); Erase to Retain.
- Novelty: medium-high for promptable foundation models. Risk: retraining from scratch, the gold-standard unlearning reference, is infeasible for a 33-dataset model, so it must be replaced by a proxy such as fine-tuning from SAM3 without BUSI. That is costly; check compute.

**RQ4 (lighter, complementary): Unlearning-aware prompt-selection protocol.** Extend the card's per-modality prompt-selection protocol with an "unlearning robustness" criterion: choose the prompt type per modality that maximises Dice while minimising residual leakage after correction. Add an unlearning-specific failure category to the taxonomy.
- Closest prior art: none found for prompt selection under unlearning; few-shot concept prompt learning (alphaxiv 2608.01663) for prompt adaptation.
- Novelty: medium. This is a protocol contribution and is best packaged as the deliverable of RQ1 rather than as a standalone paper.

- Main risks across all RQs:
  1. Verify Medical SAM3's dataset list so the "untested" claims are honest. TN3K, HC18 and PH2 are already its external sets.
  2. Point prompts are unsupported or undocumented in Medical SAM3.
  3. SCU code may not be released, which limits baseline comparison.

### Gaps
- Novelty assessments rest on searches whose full texts could not be read, because arxiv.org and ojs.aaai.org were blocked in this session. A final check on Google Scholar/arXiv is needed for "Medical SAM3 unlearning", "SAM 3 unlearning", "MedSAM unlearning" and "prompt-type unlearning".
- I did not check whether any MICCAI 2026 paper (proceedings around Sept–Oct 2026) addresses unlearning in MedSAM or SAM2. No hit came up in these searches, but MICCAI 2026 coverage in the search index may be incomplete.
