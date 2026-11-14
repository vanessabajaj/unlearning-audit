# Data

## Kvasir-SEG

Downloaded 2026-11-14 from https://datasets.simula.no/downloads/kvasir-seg.zip
- SHA-256: `03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7` (46,227,172 bytes)
- 1,000 images + 1,000 masks + `kavsir_bboxes.json`. No absolute or `..` paths in the archive.
- TLS note: the server omits its intermediate certificate (Sectigo Public Server Authentication
  CA DV R36). It was fetched from crt.sh (id 4267304690, SHA-256 fingerprint
  8C:54:C3:34:...:EF:22:E0). The chain was verified with `openssl verify` against the system root store
  before the download, which ran with verification on.

Expected layout after unzipping `kvasir-seg.zip` from https://datasets.simula.no/downloads/kvasir-seg.zip:

```
data/raw/Kvasir-SEG/images/*.jpg
data/raw/Kvasir-SEG/masks/*.jpg
data/raw/Kvasir-SEG/kavsir_bboxes.json
```

Record the archive's SHA-256 in this file once downloaded.

## Build groups and splits

```bash
.venv-power/bin/python src/dedup.py --root data/raw/Kvasir-SEG --out data/groups
```

Then look at `data/groups/review_linked.png` (weakest links that were merged) and
`data/groups/review_borderline.png` (closest pairs that were *not* merged), and check
`threshold_sweep.csv`. Adjust `--ssim-min` only on the basis of these images, before any model
is trained, and record the final thresholds here.

```bash
.venv-power/bin/python src/make_splits.py --groups data/groups/groups.csv --out data/splits --seed 20261114
```

Outputs `data/splits/splits.csv` (image_id, group_id, mask_area_frac, size_tercile, split) and
`split_summary.json`. `make_splits.py` refuses to write a split in which any group spans two partitions.

## Method (for the paper)

Near-duplicates: 64-bit DCT perceptual hash on 32x32 grayscale. Pairs are linked if the Hamming
distance is <= 4, or if it is <= 24 **and** the shift-tolerant SSIM is >= 0.75. Shift-tolerant SSIM
is the best SSIM on 64x64 grayscale over translations of up to +/-5 px.
Linked images are merged transitively into groups. Groups are assigned whole to
train_pool / trusted / val / test (65/10/10/15%), largest first, balancing polyp-area terciles.

On synthetic data (80 scenes, 221 frames with shift, zoom and brightness changes), every scene
was recovered exactly, with no merges across scenes. True duplicates scored shift-SSIM >= 0.86, and
the most similar distinct scenes scored <= 0.60. Real colonoscopy frames will differ, so the
thresholds must be checked on the review images above.

## CVC-ClinicDB (external test, not downloaded yet)

It has sequence IDs, so split or evaluate by sequence, not by frame.
