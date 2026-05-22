# IsoCourt Research: Model Training & Experiments

This document summarizes the **research program behind IsoCourt’s stroke-recognition models**: what we train on, how runs stay comparable, which architectures we built, published baselines we re-implemented, and where **RGB + skeleton fusion** fits relative to prior badminton work.

For operational training commands, see [`backend/models/MODEL_REGISTRY.md`](../backend/models/MODEL_REGISTRY.md). For TimeSformer sweeps, see [`backend/pipelines/training/TIMESFORMER_HYPERPARAMS.md`](../backend/pipelines/training/TIMESFORMER_HYPERPARAMS.md).

---

## 1. Problem & dataset

**Task:** Fine-grained **badminton stroke recognition** (and related multitask labels) from short clips around each hit in broadcast-style video.

**Primary dataset:** [FineBadminton-20K](https://huggingface.co/datasets/Moujuruo/Finebadminton-20K) (Moujuruo / FineBadminton), prepared via:

```bash
python backend/pipelines/vlm/common/prepare_finebadminton_20k.py
# optional: --extract-training-frames for 16 JPEGs per clip
```

**Annotations:** `backend/data/transformed_combined_rounds_output_en_evals_translated.json` — merged hit spans with `stroke_type`, court `position`, `technique`, `placement`, `intent`, `quality`, etc.

**Clip definition (all native IsoCourt video trainers):**

| Setting | Value |
|--------|--------|
| Frames per clip | **16** |
| Sampling | `np.linspace` over `[start_frame, end_frame)` per hit |
| Spatial size | **224×224** (ImageNet norm) for RGB models |
| Skeleton | **MediaPipe BlazePose**, **33 joints × 3** (cached) |

**Multitask heads (native trainers):** six shared tasks plus `quality` (7 levels). `stroke_subtype` is dropped in trainers to reduce head noise; **`stroke_type` (9 classes)** is the primary metric for model selection and tables below.

---

## 2. Training standards (fair comparison contract)

Every **native** video/pose trainer in `backend/pipelines/training/` is written to share the same experimental contract so numbers are comparable.

### 2.1 Split & leakage control

- **`video_level_split`** (`backend/core/split.py`): **80% / 20% by video**, `seed=42`. No clip from a validation video appears in training.
- Same split logic is reused for BST collate prep (`prepare_bst_finebadminton_collated*.py`) and VLM JSONL splits.

### 2.2 Optimization & sampling

| Item | Typical setting |
|------|-----------------|
| Batch size | **4** (`DEFAULT_TRAIN_BATCH_SIZE`) |
| Optimizer | **AdamW**, cosine warm restarts (`T_0=10`, `T_mult=2`) on heavy trainers |
| Class imbalance | **`WeightedRandomSampler`** on train, weights from train-split `stroke_type` counts |
| Primary loss emphasis | `stroke_type` weight **2.0** (video models); other tasks weight **1.0** |
| Label smoothing | **0.1** on cross-entropy |
| Augmentation | **`strong`** default (flip, affine, color jitter, erasing) — shared pattern across `train_full`, `train_conv3d`, `train_timesformer`, `train_vit_gcn` |

### 2.3 Checkpointing & logging

- **Selection metric:** best validation **`stroke_type` accuracy** (`val_type_acc`).
- **MLflow** experiment logging (hyperparameters, per-epoch train/val acc, val position acc).
- **Registry:** `backend/models/model_registry.json` stores `primary` checkpoints per architecture category.

### 2.4 Pose cache (shared skeleton signal)

- Built once: `backend/models/pose_cache_mediapipe.pt` → tensor **`(N, 16, 33, 3)`** aligned to dataset sample index `N`.
- If `pose_cache_mediapipe.pt` is missing, an older on-disk cache filename in the same folder may still be loaded.
- Video models accept **`--no-pose`** for RGB-only ablations without rebuilding the cache.

### 2.5 Skeleton-only baselines (different I/O)

BST / TemPose / ST-GCN use **pre-collated** `.npy` under `backend/data/bst_finebadminton_collated/`:

| Property | Native IsoCourt | BST collated | TemPose_V / ST-GCN |
|----------|-----------------|--------------|---------------------|
| Joints | MediaPipe **33** | COCO **17** (+ optional bones) | COCO **17** |
| Sequence length | **16** | often **30** (BST default) | configurable (e.g. 16) |
| Extra inputs | RGB frames | shuttle + court position | skeleton only (TemPose_V) |
| Task | Multitask | **stroke_type only** | **stroke_type only** |

When comparing to native models, treat these as **external references** on the same label file and video split, not identical tensors.

---

## 3. RGB + skeleton fusion (core design theme)

IsoCourt’s main research thread is **fusing appearance (video) with body kinematics (skeleton)** under one training contract. Fusion is **not one monolithic module** — we implement several fusion points so ablations are meaningful:

| Architecture | Fusion style | Where skeleton meets RGB |
|--------------|--------------|---------------------------|
| **CNN + LSTM** | **Early** | Flattened pose (**99-d**) concatenated to ResNet50 features **before** LSTM |
| **Conv3D + pose** | **Late** | R(2+1)D trunk features ∥ pose MLP → multitask heads |
| **TimeSformer + pose** | **Token** | One **pose token** prepended to patch tokens **each frame**, then divided space–time blocks |
| **ViT + GCN** | **Dual-stream** | Per-frame ViT CLS ∥ per-frame GCN on 33 joints → concat → fusion MLP |
| **ST-TR + ViT** | **Late** | Official ST-TR skeleton embedding ∥ pooled ViT clip embedding → fusion MLP |

**Skeleton-only** paths (no RGB): custom ST-TR, vendored **SkateFormer**, and collated **BST / TemPose / ST-GCN**.

### Is “Image + Skeleton Fusion” our main novel contribution?

**Partially — with important nuance.**

- **Not novel in the abstract:** RGB–pose fusion is well established in action recognition; badminton already has strong **skeleton-first** systems (**BST**, **TemPose**) that use pose, shuttle, and court context.
- **What IsoCourt adds that is research-relevant:**
  1. **A unified FineBadminton-20K benchmark** — same 16-frame hit clips, video-level split, MediaPipe cache, and multitask heads across heterogeneous backbones (CNN-LSTM → 3D CNN → divided ST → ViT-GCN).
  2. **Systematic pose ablations** via `--no-pose` on every RGB-capable native model (same cache, same split).
  3. **Multiple fusion mechanisms** compared under identical data (early, late, token, dual-stream) — not only one fusion design.
  4. **End-to-end product coupling:** sliding-window inference, skeleton overlay, and **LLM coaching** on structured model outputs (deployment story beyond a single accuracy table).
  5. **Adaptation of recent general skeleton models** (e.g. SkateFormer ECCV 2024) to badminton with MediaPipe topology — transfer-learning narrative.

The strongest **empirical** claim today is: **late-fused 3D CNN + MediaPipe pose beats RGB-only CNN-LSTM on `stroke_type` val acc** (see §4). Token-level TimeSformer (scratch) has **not** yet beaten the CNN-LSTM baseline without ViT pretraining and tuning.

---

## 4. Main results table (stroke_type validation accuracy)

Metric: **best validation `stroke_type` accuracy (%)** on the standard video-level split, unless noted.  
**Pose column:** whether the registered primary checkpoint uses skeleton fusion (`inference.use_pose` in registry).

| Model | Modality | Pose | Val acc (%) | Registry / status |
|-------|----------|------|-------------|-------------------|
| **CNN + LSTM** | RGB (+ optional early pose) | **No** (primary) | **68.71** | `cnn_lstm` — production-style RGB baseline |
| **CNN + LSTM** | RGB + early pose | *TBD* | — | Run: `train_full.py` (default pose on; compare to `--no-pose`) |
| **Conv3D + pose** (R(2+1)D-18) | RGB + late pose | **Yes** | **71.31** | `conv3d_pose` — **best native checkpoint to date** |
| **Conv3D** | RGB only | *TBD* | — | Run: `train_conv3d.py --no-pose` |
| **TimeSformer (scratch)** | RGB + pose token | **Yes** | **53.49** | `timesformer` — conv patch stem, needs more tuning |
| **TimeSformer (ViT)** | RGB + pose token | *TBD* | — | Run: `--backbone vit --vit-model vit_small_patch16_224` (+ unfreeze sweeps in TIMESFORMER_HYPERPARAMS.md) |
| **ViT + GCN** | RGB + skeleton graph | *TBD* | — | `vit_gcn` — no `primary` in registry yet |
| **ViT + GCN** | RGB only | *TBD* | — | Run: `train_vit_gcn.py --no-pose` |
| **SkateFormer-B** | Skeleton only (MediaPipe 33) | N/A | *In progress* | Official backbone in `core/skateformer/`; training referenced in `logs/` (see §5.6) |
| **ST-TR** (custom) | Skeleton only | N/A | **10.83** | `st_tr` — under-trained / poor fit; not used in API |
| **ST-TR + ViT** | RGB + skeleton (late) | Yes | *TBD* | `train_st_tr_vit.py` — experiment path |

> **Update this table** after each MLflow run: copy best `val_type_acc` into `model_registry.json` or your lab spreadsheet. Rows marked *TBD* are the **pose vs no-pose ablation** cells you should fill next.

### 4.1 Pose vs no-pose (ablation protocol)

For any row with RGB:

```bash
# With pose (default)
python backend/pipelines/training/train_<arch>.py

# Without pose
python backend/pipelines/training/train_<arch>.py --no-pose
```

Keep **seed, split, epochs, batch size, and aug** fixed; only toggle `--no-pose`. Expect pose to help most when (a) stroke classes are kinematically distinct (smash vs drop) and (b) RGB is noisy (motion blur, similar backgrounds).

---

## 5. Architectures (native IsoCourt)

### 5.1 CNN + LSTM (`train_full.py` → `core/model.py`)

- **RGB:** ResNet50 per frame → **2048-d** features.
- **Temporal:** 1-layer LSTM (`hidden_size` typically 128 in registry).
- **Pooling:** temporal **avg + max** → concat → multitask MLP heads.
- **Pose (optional):** `use_pose=True` concatenates **99-d** pose vector per timestep into LSTM input.
- **Current primary:** trained **without pose** (`use_pose: false`, 68.71% val) — still the default API category in many deployments.

### 5.2 Conv3D + pose (`train_conv3d.py` → `core/conv3d_pose.py`)

- **RGB:** torchvision **R(2+1)D-18** (Kinetics-pretrained), spatial **224**, 16 frames.
- **Training:** freeze trunk; unfreeze **layer4** + fusion/heads (differential LR).
- **Pose:** flatten `(T,33,3)` → linear → **late concat** with video embedding.
- **Why it leads:** 3D spatiotemporal conv captures racket motion in pixels; pose disambiguates technique when appearance is ambiguous.

### 5.3 TimeSformer — scratch (`train_timesformer.py` → `core/timesformer.py`, `backbone=scratch`)

- **RGB:** conv patch embed (16×16 on 224).
- **ST:** stacked **divided** spatial-then-temporal transformer blocks.
- **Pose:** linear projection of 33×3 → **one extra token per frame** before spatial attention.
- **Status:** registry primary at **53.49%** — scratch stem is data-hungry; use ViT backbone + sweeps before drawing final conclusions.

### 5.4 TimeSformer — ViT (`--backbone vit`)

- **RGB:** timm ViT patch tokens (ImageNet pretrained); CLS discarded; pose token prepended like scratch path.
- **Tuning:** `--vit-unfreeze-last-n`, `--vit-lr-mult`, stroke loss weight, aug strength — see TIMESFORMER_HYPERPARAMS.md.

### 5.5 ViT + GCN (`train_vit_gcn.py` → `core/vit_gcn.py`)

- **RGB stream:** timm ViT **CLS** per frame → project → temporal avg/max.
- **Skeleton stream:** **fixed adjacency GCN** on MediaPipe 33 graph → per-frame pool → temporal avg/max.
- **Fusion:** concat **4×embed_dim** → MLP → heads (or **2×embed_dim** RGB-only with `--no-pose`).
- **Role in research:** explicit **graph structure** on skeleton vs unstructured pose tokens in TimeSformer.

### 5.6 SkateFormer-B (skeleton-only)

- **Backbone:** ECCV 2024 [SkateFormer](https://github.com/KAIST-VICLab/SkateFormer) vendored in `backend/core/skateformer/official.py` — partitioned skeletal–temporal attention.
- **IsoCourt adapter:** MediaPipe **33** joints, **T=16**, hip-centered normalization (same cache as ST-TR).
- **“B” configuration:** compact training profile used on GPU runs (e.g. `embed_dim=64`, `num_heads=16`, batch 4) — distinguish from full NTU-scale configs in the upstream repo.
- **Status:** integration and `train_skateformer.py` are active in development; check `logs/train-skateformer-*.log` and MLflow for latest val acc before promoting to registry.

### 5.7 ST-TR + ViT (`train_st_tr_vit.py` → `core/st_tr_vit_fusion.py`)

- **Skeleton:** upstream **ST-TR** on BlazePose graph (`core/st_tr_official.py`).
- **RGB:** fine-tuned **ViT** clip encoder (late fusion).
- **Training recipe:** optional `--stroke-only-epochs`, skeleton warm-start from `badminton_model_st_tr.pth`, medium/strong aug.
- **Purpose:** test whether **late fusion** of a stronger skeleton trunk + ViT beats pure pose tokens in TimeSformer.

---

## 6. Other models tested (baselines & references)

These live under `backend/third_party/` and separate training scripts. They answer: *“How do published badminton skeleton systems compare on our labels?”*

| Model | Script | Inputs | Notes |
|-------|--------|--------|-------|
| **BST** | `train_bst_baseline.py` | COCO-17 pose (+ bones), **shuttle**, **court position**, video length | MIT [BST](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer); default **BST_CG_AP**, T=30 collate |
| **TemPose_V** | `train_tempose_baseline.py` | Skeleton **only** (joints + bones) | Fairer “pose-only SOTA” reference than full BST |
| **ST-GCN** | `train_stgcn_baseline.py` | COCO-17 joints, 2 players × 17 | Classic graph conv baseline; no shuttle |
| **SkateFormer** (upstream) | BST repo `skateformer_main.py` / IsoCourt `core/skateformer` | NTU-style skeleton | General HAR SOTA transfer target for badminton |
| **ST-TR** (custom) | `core/st_tr.py` (trainer referenced in registry) | MediaPipe 33, dual transformer | Poor val acc; superseded by official ST-TR + SkateFormer experiments |

**Prepare collated tensors (once):**

```bash
python backend/pipelines/training/prepare_bst_finebadminton_collated.py \
  --output-dir backend/data/bst_finebadminton_collated
# or MMPose variant:
python backend/pipelines/training/prepare_bst_finebadminton_collated_mmpose.py
```

---

## 7. Possible novel contributions (paper / thesis bullets)

Use these as **claims to support with tables and ablations**, not as established literature:

1. **Unified multimodal benchmark on FineBadminton-20K** — comparable 16-frame clips, video-level split, and multitask evaluation across CNN, 3D CNN, divided space–time transformers, ViT-GCN, and skeleton transformers.
2. **Fusion-point study for badminton** — early (LSTM), late (Conv3D), token (TimeSformer), and graph dual-stream (ViT-GCN) under shared pose cache and metrics.
3. **Empirical finding (current):** **late fusion with 3D CNN + MediaPipe** outperforms **RGB-only ResNet-LSTM** on stroke_type (+2.6 pp val acc in registry) — supports kinematics when RGB alone plateaus.
4. **Transfer of ECCV 2024 SkateFormer** (and related skeleton transformers) to **racket sports** with BlazePose topology — gap in published badminton literature.
5. **Production-oriented HAR:** coupling stroke models with **real-time sliding windows**, skeleton visualization, and **LLM-generated coaching** from structured predictions (system paper angle).
6. **Negative result / method note:** vanilla **ST-TR without graph structure** underperforms on fine-grained strokes — motivates graph/partition models (GCN, SkateFormer, BlockGCN-style follow-ups).

**What is weaker as a novelty claim:** “We invented image + skeleton fusion.” Prior work (including BST/TemPose for badminton, and generic two-stream HAR) already combines modalities. Frame the contribution as **rigorous comparison + badminton-specific benchmark + deployment**, not fusion itself.

---

## 8. Recommended experiment backlog

| Priority | Experiment | Command hint |
|----------|------------|--------------|
| High | Pose ablation on **Conv3D** | `train_conv3d.py --no-pose` vs default |
| High | **TimeSformer ViT** + unfreeze sweep | `train_timesformer.py --backbone vit` + TIMESFORMER_HYPERPARAMS.md |
| High | **CNN+LSTM with pose** | `train_full.py` (pose on) vs current RGB-only primary |
| Medium | **ViT-GCN** pose on/off | `train_vit_gcn.py` / `--no-pose` |
| Medium | **TemPose_V** vs **SkateFormer** on same collate/split | baseline scripts + SkateFormer trainer |
| Medium | Promote best checkpoint | `python -m api.inference_model_cli set <category>` |
| Low | ST-TR+ViT full multitask after stroke-only warmup | `train_st_tr_vit.py` |

---

## 9. File map (quick reference)

| Topic | Path |
|-------|------|
| Dataset & labels | `backend/core/dataset.py`, `backend/core/finebadminton_dataset_spec.py` |
| Split | `backend/core/split.py` |
| Pose cache | `backend/core/pose_cache_build.py` |
| Models | `backend/core/model.py`, `conv3d_pose.py`, `timesformer.py`, `vit_gcn.py`, `st_tr.py`, `st_tr_vit_fusion.py`, `skateformer/` |
| Training | `backend/pipelines/training/train_*.py` |
| Registry | `backend/models/model_registry.json`, `MODEL_REGISTRY.md` |
| Baselines | `backend/third_party/BST-*`, `backend/third_party/ST-TR/` |

---

*Last synced with registry checkpoints: May 2026. Update §4 when new primaries are saved.*
