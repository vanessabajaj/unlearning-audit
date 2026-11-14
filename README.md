# Did It Really Forget? Spatial residual-influence auditing of corrective unlearning

Research code for auditing whether corrective unlearning in a promptable segmentation model
(SAM ViT-B on Kvasir-SEG) removes the influence of deliberately corrupted masks, and
whether a spatial internal audit predicts re-emergence under prompt perturbation and fine-tuning.

| Path | Contents |
|---|---|
| `design/DESIGN.md` | Study design: conditions, partitions, audit controls, recovery arms, pre-specified analysis |
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
