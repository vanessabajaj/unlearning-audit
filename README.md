# Does the fix hold? Prompt-conditional corrective unlearning in Medical SAM3

Research code for assignment card A ("Does promptable medical segmentation transfer across
modalities?", base paper Medical SAM3, arXiv:2601.10880), with a corrective-unlearning twist.
After unlearning corrupted masks in Medical SAM3 fine-tuned on held-out ultrasound and dermoscopy
data, does the fix hold across prompt types (text, box, point) and modalities? And does a
pixel-level internal audit predict where errors re-emerge?

The layout and power files in `design/` still describe design v1 (SAM ViT-B on Kvasir-SEG) and
are due to be regenerated; see `design/DESIGN.md` Sec. 10.

| Path | Contents |
|---|---|
| `design/DESIGN.md` | Study design v2 (Medical SAM3, held-out ultrasound and dermoscopy, prompt-conditional unlearning) |
| `design/make_layout.py` | Seeded run manifest (`runs.csv`), prompt grid, fine-tuning checkpoints |
| `design/power_sim*.py` | Simulation-based power analysis behind the seed count |
| `src/dedup.py` | Near-duplicate grouping (perceptual hash + shift-tolerant SSIM) |
| `src/make_splits.py` | Group-wise, size-stratified train/trusted/val/test split |
| `tests/` | Synthetic tests for grouping and splitting |
| `refs/references.bib` | Verified references |
| `data/README.md` | How to obtain the data, plus its checksum and provenance |

## Setup

```bash
uv venv --python 3.13 .venv-power
uv pip install --python .venv-power/bin/python "statsmodels==0.15.0" "scipy==1.18.1" "numpy==2.5.3" "pandas==2.3.3" "matplotlib==3.11.2" pillow pytest
.venv-power/bin/python -m pytest -q tests
```

The dataset is not included. See `data/README.md`.
