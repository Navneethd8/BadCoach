# Cluster retrain runbook (tmux)

Copy-paste commands for **nd17** (or any cluster node) after code is synced and data is symlinked.

Run everything from **repo root** (`~/IsoCourt`):

```bash
cd ~/IsoCourt
chmod +x scripts/cluster/*.sh
```

See also: [`README.md`](README.md) (bootstrap, rsync), [`RETRAIN_TMUX_LOG.md`](RETRAIN_TMUX_LOG.md) (paste what you ran), [`../models/MODEL_REGISTRY.md`](../../backend/models/MODEL_REGISTRY.md) (checkpoints).

**Quick rule:** runnable commands below use **literal paths** (no `"${DATA_ROOT}"` traps). You still need §1 **GPU / tmux / python** exports—or inline `export ISOCOURT_TMUX_SESSION=...` in each block.

**Common paths (copy reference):**

| What | Path |
| ---- | ---- |
| Data root | `backend/data` |
| Labels JSON | `backend/data/transformed_combined_rounds_output_en_evals_translated.json` |
| Pose cache (all trainers + VLM pose) | `backend/models/pose_cache_mediapipe.pt` |
| BST collate MMPose (T=16, **use for §4**) | `backend/data/bst_finebadminton_collated_mmpose_16` |
| BST collate MediaPipe (legacy; sparse pose) | `backend/data/bst_finebadminton_collated_16` |
| VLM JSONL | `backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl` |

---

## 0. One-time setup (once per machine)

```bash
./scripts/cluster/setup_data_symlink.sh
./scripts/cluster/bootstrap_cluster.sh

# Conda example (adjust path):
conda activate isocourt
```

---

## 1. Shared tmux + GPU exports

Paste once per SSH session. **Path variables are optional**—commands in §2–§7 use literal paths. You mainly need python + GPU + tmux settings here.

```bash
# --- Repo ---
cd ~/IsoCourt

# --- Python (pick one) ---
export ISOCOURT_VENV="$HOME/miniconda3/envs/isocourt"
export ISOCOURT_PYTHON="${ISOCOURT_PYTHON:-${ISOCOURT_VENV}/bin/python}"

# --- GPU / tmux ---
export CUDA_VISIBLE_DEVICES=0
export ISOCOURT_TMUX_REPLACE=1          # kill existing session name before start
export ISOCOURT_TMUX_SESSION=isocourt-train
export ISOCOURT_DISABLE_MLFLOW=1        # set 0 to log to backend/mlruns

# --- Optional: rsync checkpoints to laptop after successful train ---
# export ISOCOURT_RSYNC_DEST="you@laptop:~/IsoCourt/"
# export ISOCOURT_RSYNC_EXTRA='--progress'

# --- OpenAI only (§7) ---
# export OPENAI_API_KEY="sk-..."
# export OPENAI_VLM_MODEL="gpt-5.5"
# export OPENAI_VLM_REASONING_EFFORT="none"
```

**Parallel jobs:** use a different GPU and session per job:

```bash
export CUDA_VISIBLE_DEVICES=1
export ISOCOURT_TMUX_SESSION=isocourt-jvc
export ISOCOURT_TMUX_REPLACE=1
```

**Attach / detach:**

```bash
tmux attach -t "${ISOCOURT_TMUX_SESSION}"
# detach: Ctrl-b then d
```

**Registry default:** trainers save to a **timestamped** `.pth` and append to `registrations`. Pass `--no-registry-experiment` only when you intentionally want to overwrite `primary`.

---

## 2. Prerequisites (before model training)

### VLM JSONL (Qwen + OpenAI eval)

```bash
./scripts/cluster/prepare_vlm_16frame.sh
# Qwen/OpenAI pose mode uses backend/models/pose_cache_mediapipe.pt
```

### BST collate — MMPose (recommended; GPU; run once at T=16)

Use for **TemPose-V**, **ST-GCN**, and skeleton features in §4. Matches the BST paper pose pipeline (RTMPose at native resolution). Output layout is identical to MediaPipe collate.

**Do not install MMPose into `isocourt`.** That env uses torch 2.10+cu128, which has no reliable prebuilt `mmcv` wheels (and source builds often fail). Use a **separate conda env** for collate prep only:

```bash
cd ~/IsoCourt
chmod +x scripts/cluster/install_bst_mmpose.sh
./scripts/cluster/install_bst_mmpose.sh
```

This creates `isocourt-mmpose` (python 3.10, torch 2.3.1+cu121 — same stack as the BST paper). Verify:

```bash
conda activate isocourt-mmpose
python -c "from mmpose.apis import MMPoseInferencer; print('mmpose OK')"
```

(`Skipping import of cpp extensions due to incompatible torch version` is a harmless mmcv warning if the import succeeds.)

**Use a GPU** — much faster than CPU (~1–2 h vs many hours on a T4-class card).

**While collate runs:** you can start §3 native trainers (and VLM prep) in other tmux sessions—they do not need the collated `.npy`. Wait for collate before §4 (BST / TemPose / ST-GCN).

Quick sanity check on cluster:

```bash
ls -la backend/data
test -f backend/data/transformed_combined_rounds_output_en_evals_translated.json && echo "labels OK"
conda activate isocourt-mmpose
python -c "from mmpose.apis import MMPoseInferencer; print('mmpose OK')"
```

Run prep with the mmpose env python (output `.npy` works in `isocourt` for training).

**Headless OpenCV + numpy pin (required on nd17):** after any `pip install` in this env:

```bash
conda activate isocourt-mmpose
chmod +x scripts/cluster/ensure_mmpose_env.sh
./scripts/cluster/ensure_mmpose_env.sh
```

```bash
export ISOCOURT_PYTHON="$HOME/miniconda3/envs/isocourt-mmpose/bin/python"
export CUDA_VISIBLE_DEVICES=0
export ISOCOURT_TMUX_SESSION=isocourt-bst-prep-mmpose-16
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh bst_prep_mmpose \
  --data-root backend/data \
  --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \
  --output-dir backend/data/bst_finebadminton_collated_mmpose_16 \
  --sequence-length 16 \
  --pose-style JnB_bone
```

Prep **auto-uses** training JPEGs at `backend/data/FineBadminton-20K/dataset/image/` when present (same `{stem}_{frame}.jpg` as `FineBadmintonDataset`). Explicit path or native video:

```bash
  --image-dir backend/data/FineBadminton-20K/dataset/image   # optional; same as auto-detect
  --prefer-video                                              # slow: mp4 seek @ native res
```

### BST collate — MediaPipe Lite (legacy; not for skeleton baselines)

MediaPipe at 224×224 produces sparse dual-player pose; TemPose/ST-GCN tend to stick at ~majority-class val acc. Keep only for debugging or comparison.

```bash
export ISOCOURT_TMUX_SESSION=isocourt-bst-prep-16
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh bst_prep \
  --data-root backend/data \
  --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \
  --output-dir backend/data/bst_finebadminton_collated_16 \
  --sequence-length 16 \
  --pose-style JnB_bone
```

---

## 3. Native pose models

Can run **in parallel** with §2 BST collate (different tmux session + GPU). Does not need collated `.npy`.

### CNN+LSTM (`full` / `cnn_lstm`)

```bash
export ISOCOURT_TMUX_SESSION=isocourt-cnn-lstm
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh full \
  --data-root backend/data \
  --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \
  --pose-cache-path backend/models/pose_cache_mediapipe.pt \
  --epochs 60 \
  --batch-size 4
```

### Conv3D + pose (`conv3d_pose`)

```bash
export ISOCOURT_TMUX_SESSION=isocourt-conv3d
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh conv3d_pose \
  --pose-cache backend/models/pose_cache_mediapipe.pt \
  --epochs 60 \
  --batch-size 4 \
  --video-backbone r2plus1d_18 \
  --spatial-size 224 \
  --aug strong
```

### TimeSformer + pose (`timesformer`)

ViT per-frame backbone (ImageNet `vit_small_patch16_224` @ 224) + divided space–time blocks + pose tokens.

```bash
export ISOCOURT_TMUX_SESSION=isocourt-timesformer
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh timesformer \
  --pose-cache backend/models/pose_cache_mediapipe.pt \
  --epochs 60 \
  --batch-size 4 \
  --backbone vit \
  --vit-model vit_small_patch16_224 \
  --aug strong
```

Optional: unfreeze last ViT blocks after val plateaus: `--vit-unfreeze-last-n 2` (uses `--vit-lr-mult 0.25` by default).

### JVC (`jvc`)

```bash
export ISOCOURT_TMUX_SESSION=isocourt-jvc
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh jvc \
  --pose-cache backend/models/pose_cache_mediapipe.pt \
  --vision-backbone conv3d \
  --sampling hit_centered \
  --epochs 60 \
  --batch-size 4 \
  --stroke-only-epochs 4
```

Optional warm-start (if prior checkpoints exist on cluster):

```bash
  --resume-checkpoint backend/models/badminton_model_conv3d_pose.pth \
  --resume-jvc backend/models/badminton_model_jvc.pth
```

---

## 4. External skeleton baselines

Requires **MMPose collate** from §2 (`bst_finebadminton_collated_mmpose_16`).

### BST

```bash
export ISOCOURT_TMUX_SESSION=isocourt-bst
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh bst \
  --collated-root backend/data/bst_finebadminton_collated_mmpose_16 \
  --sequence-length 16 \
  --pose-style JnB_bone \
  --model-name BST_CG_AP \
  --epochs 60 \
  --batch-size 64
```

### TemPose-V (`tempose`)

```bash
export ISOCOURT_TMUX_SESSION=isocourt-tempose
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh tempose \
  --collated-root backend/data/bst_finebadminton_collated_mmpose_16 \
  --sequence-length 16 \
  --pose-style JnB_bone \
  --epochs 60 \
  --batch-size 64
```

### ST-GCN (`stgcn`)

```bash
export ISOCOURT_TMUX_SESSION=isocourt-stgcn
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh stgcn \
  --collated-root backend/data/bst_finebadminton_collated_mmpose_16 \
  --sequence-length 16 \
  --pose-style JnB_bone \
  --epochs 60 \
  --batch-size 64
```

### Test eval (after training — no MLflow)

Runs `backend/scripts/eval_skeleton_baseline_checkpoint.py` on the collated **test** split only (not val). Default checkpoint: `backend/models/badminton_model_{bst,tempose,stgcn}.pth`.

```bash
unset ISOCOURT_PYTHON
export ISOCOURT_VENV="$HOME/miniconda3/envs/isocourt"
export ISOCOURT_COLLATED_ROOT=backend/data/bst_finebadminton_collated_mmpose_16

# BST test accuracy
export ISOCOURT_TMUX_SESSION=isocourt-bst-test
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_eval_tmux.sh bst --per-class
# override checkpoint if training saved a timestamped .pth:
# export ISOCOURT_CHECKPOINT=backend/models/badminton_model_bst_20260715T120000Z.pth

# TemPose test
export ISOCOURT_TMUX_SESSION=isocourt-tempose-test
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_eval_tmux.sh tempose --per-class

# ST-GCN test
export ISOCOURT_TMUX_SESSION=isocourt-stgcn-test
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_eval_tmux.sh stgcn --per-class
```

Logs: `logs/eval-{model}-*.log`. Pass `--split val` only if you explicitly want val (default is always `test`).

### Native baselines — test eval (CNN-LSTM, Conv3D, TimeSformer, JVC)

Uses `backend/scripts/eval_native_baseline_checkpoint.py` on the **video-level test split** (14 test videos) with `pose_cache_mediapipe.pt`. No MLflow.

```bash
unset ISOCOURT_PYTHON
export ISOCOURT_VENV="$HOME/miniconda3/envs/isocourt"
export CUDA_VISIBLE_DEVICES=3
export ISOCOURT_TMUX_REPLACE=1

# CNN-LSTM (train_full)
export ISOCOURT_CHECKPOINT=backend/models/badminton_model_20260714T030922Z.pth
export ISOCOURT_TMUX_SESSION=isocourt-cnn-lstm-test
./scripts/cluster/run_eval_tmux.sh cnn_lstm --per-class

# Conv3D + pose
export ISOCOURT_CHECKPOINT=backend/models/badminton_model_conv3d_pose_20260714T030509Z.pth
export ISOCOURT_TMUX_SESSION=isocourt-conv3d-test
./scripts/cluster/run_eval_tmux.sh conv3d --per-class

# TimeSformer
export ISOCOURT_CHECKPOINT=backend/models/badminton_model_timesformer_20260714T073004Z.pth
export ISOCOURT_TMUX_SESSION=isocourt-timesformer-test
./scripts/cluster/run_eval_tmux.sh timesformer --per-class

# JVC (set your latest badminton_model_jvc_*.pth)
export ISOCOURT_CHECKPOINT=backend/models/badminton_model_jvc_YYYYMMDDTHHMMSSZ.pth
export ISOCOURT_TMUX_SESSION=isocourt-jvc-test
./scripts/cluster/run_eval_tmux.sh jvc --per-class
```

Optional: `--tta-flip` for Conv3D / TimeSformer / JVC (horizontal flip TTA on RGB).

---

## 5. JVC no-cross-attn ablations

Uses the shared MediaPipe pose cache (`pose_cache_mediapipe.pt`).

### 4-stream (default)

```bash
export ISOCOURT_TMUX_SESSION=isocourt-jvc-noxattn-4s
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh jvc_no_xattn \
  --pose-cache backend/models/pose_cache_mediapipe.pt \
  --epochs 60 \
  --batch-size 4
```

### 1-stream

```bash
export ISOCOURT_TMUX_SESSION=isocourt-jvc-noxattn-1s
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh jvc_no_xattn \
  --pose-cache backend/models/pose_cache_mediapipe.pt \
  --no-four-stream \
  --epochs 60 \
  --batch-size 4
```

Optional warm-start:

```bash
  --resume-skeleton backend/models/badminton_model_skateformer_b.pth \
  --resume-checkpoint backend/models/badminton_model_conv3d_pose.pth
```

---

## 6. Qwen3-VL-8B (train)

Install Unsloth stack once on GPU node:

```bash
pip install -r backend/pipelines/vlm/common/requirements-unsloth-vlm.txt
```

### All `train_qwen3_vl_8b.py` flags (via tmux)

| Flag | Default | Notes |
| ---- | ------- | ----- |
| `--jsonl` | *(required)* | 16-frame JSONL from `prepare_vlm_16frame.sh` |
| `--model_name` | Qwen3-VL-8B-Instruct | HF model id |
| `--output_dir` | `outputs/qwen3_vl_8b_lora` | LoRA + tokenizer saved here |
| `--pose_mode` | `cache_text` | **`none`** = no-pose ablation; also `overlay`, `text`, `both` |
| `--pose_cache_path` | auto | `backend/models/pose_cache_mediapipe.pt` for `cache_text` |
| `--pose_model_path` | auto | Live MediaPipe (if not using cache) |
| `--pose_min_short_edge` | `960` | Set `0` to disable upscale before pose |
| `--num_frames` | `16` | |
| `--frame_size` | `224` | |
| `--max_pixels_per_image` | project default | Lower if OOM |
| `--num_train_epochs` | `5` | Or use `--max_steps` instead |
| `--max_steps` | off | Overrides epochs when set |
| `--per_device_train_batch_size` | `1` | Use `2` on H100 |
| `--per_device_eval_batch_size` | `1` | |
| `--gradient_accumulation_steps` | `8` | |
| `--learning_rate` | `2e-4` | |
| `--warmup_steps` | `5` | |
| `--max_seq_length` | config default | |
| `--load_in_4bit` / `--no-load_in_4bit` | `True` | |
| `--finetune_vision` / `--no-finetune_vision` | `True` | |
| `--finetune_language` / `--no-finetune_language` | `True` | |
| `--r`, `--lora_alpha` | `16`, `16` | LoRA rank |
| `--gradient_checkpointing` | `unsloth` | |
| `--split_seed` | `42` | Video-level **70/10/20** split (49/7/14 videos on 70-match corpus) |
| `--no_val_split` | off | Train on all rows (no eval) |
| `--max_eval_samples` | `500` | Cap val eval size |
| `--dataloader_num_workers` | `0` | |
| `--save_total_limit` | `3` | |
| `--logging_steps` | `1` | |
| `--seed` | `3407` | |
| `--report_to` | `none` | e.g. `wandb` if configured |

### Qwen + pose (`cache_text`)

```bash
export ISOCOURT_TMUX_SESSION=isocourt-qwen-pose
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh qwen3_vl_8b \
  --jsonl backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl \
  --pose_mode cache_text \
  --pose_cache_path backend/models/pose_cache_mediapipe.pt \
  --num_frames 16 \
  --frame_size 224 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --split_seed 42 \
  --max_eval_samples 500 \
  --output_dir backend/pipelines/vlm/qwen-8b/outputs/qwen3_vl_8b_16frame_pose_lora
```

### Qwen no-pose ablation

```bash
export ISOCOURT_TMUX_SESSION=isocourt-qwen-nopose
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh qwen3_vl_8b \
  --jsonl backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl \
  --pose_mode none \
  --num_frames 16 \
  --frame_size 224 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --output_dir backend/pipelines/vlm/qwen-8b/outputs/qwen3_vl_8b_16frame_nopose_lora
```

### Qwen post-train test eval (optional)

After train finishes (test split — run sanity check in §7 for clip count):

```bash
"${ISOCOURT_PYTHON}" backend/scripts/eval_vlm_stroke_checkpoint.py \
  --jsonl backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl \
  --split test \
  --model_path backend/pipelines/vlm/qwen-8b/outputs/qwen3_vl_8b_16frame_pose_lora/lora_adapter \
  --pose_mode cache_text \
  --pose_cache_path backend/models/pose_cache_mediapipe.pt \
  --num_frames 16 \
  --frame_size 224 \
  --prompt_mode classify
```

---

## 7. OpenAI (eval only — not tmux training)

Zero-shot API eval on the **test split** (video-level 70/10/20). Requires `OPENAI_API_KEY` and prepared JSONL (§2). **No gradient training.**

There is **no** `--no-reasoning` flag. Reasoning is controlled by **`--reasoning_effort`** (or env `OPENAI_VLM_REASONING_EFFORT`). For one-line stroke classification on gpt-5.x, use **`none`** (already the default).

### Session setup (paste once)

```bash
cd ~/IsoCourt
mkdir -p logs backend/pipelines/vlm/openai/outputs

export ISOCOURT_VENV="$HOME/miniconda3/envs/isocourt"
export ISOCOURT_PYTHON="${ISOCOURT_PYTHON:-${ISOCOURT_VENV}/bin/python}"

export OPENAI_API_KEY="sk-..."          # your key
export OPENAI_VLM_MODEL="gpt-5.5"
export OPENAI_VLM_REASONING_EFFORT="none"
export PYTHONPATH="${PWD}/backend:${PYTHONPATH:-}"
```

### Sanity check (optional — expect **49 / 7 / 14 videos** on FineBadminton-20K)

```bash
"${ISOCOURT_PYTHON}" -c "
import json
from pathlib import Path
from core.split import vlm_jsonl_video_level_split
rows = [json.loads(l) for l in Path('backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl').open() if l.strip()]
t, v, te = vlm_jsonl_video_level_split(rows, image_key='images')
print('train', len(t), 'val', len(v), 'test', len(te))
"
```

Expected video counts: **49 train / 7 val / 14 test** (70 matches total). Clip counts vary — note the printed `test` row count for OpenAI/Qwen eval.

### All `eval_openai_stroke.py` flags

| Flag | Default | Notes |
| ---- | ------- | ----- |
| `--jsonl` | *(required)* | |
| `--split` | **`test`** | `test` \| `val` \| `train` — use **`test`** for paper numbers |
| `--split_seed` | `42` | Match native split |
| `--model` | `gpt-5.5` | Override via `OPENAI_VLM_MODEL` |
| `--pose_mode` | `cache_text` | **`none`** = no-pose ablation |
| `--pose_cache_path` | auto | `backend/models/pose_cache_mediapipe.pt` when `cache_text` |
| `--num_frames` | `16` | |
| `--frame_size` | `224` | |
| `--prompt_mode` | `classify` | `jsonl` = legacy open caption in JSONL |
| `--reasoning_effort` | **`none`** | **`none`** \| `minimal` \| `low` — use `none` for classify |
| `--max_completion_tokens` | `2048` | Bump to `4096` if gpt-5.x returns empty text |
| `--max_samples` | all split rows | Smoke test with e.g. `50` |
| `--cache_path` | off | Resume file (JSONL of predictions) |
| `--resume` | off | Skip rows already in `--cache_path` |
| `--dump_samples` | `0` | Print first N `(gt, pred, raw)` triples |

Env vars (alternative to CLI):

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_VLM_MODEL="gpt-5.5"              # or gpt-4o if gpt-5.x misbehaves
export OPENAI_VLM_REASONING_EFFORT="none"      # same as --reasoning_effort none
export PYTHONPATH="${PWD}/backend:${PYTHONPATH:-}"
```

### OpenAI + pose — test split (foreground)

**First run:** omit `--resume`. **After interrupt:** add `--resume`.

```bash
"${ISOCOURT_PYTHON}" backend/pipelines/vlm/openai/eval_openai_stroke.py \
  --jsonl backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl \
  --split test \
  --model gpt-5.5 \
  --pose_mode cache_text \
  --pose_cache_path backend/models/pose_cache_mediapipe.pt \
  --num_frames 16 \
  --frame_size 224 \
  --prompt_mode classify \
  --reasoning_effort none \
  --max_completion_tokens 4096 \
  --cache_path backend/pipelines/vlm/openai/outputs/openai_test_pose_cache.jsonl \
  2>&1 | tee logs/openai-eval-test-pose.log
```

Resume after interrupt:

```bash
"${ISOCOURT_PYTHON}" backend/pipelines/vlm/openai/eval_openai_stroke.py \
  --jsonl backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl \
  --split test \
  --model gpt-5.5 \
  --pose_mode cache_text \
  --pose_cache_path backend/models/pose_cache_mediapipe.pt \
  --num_frames 16 \
  --frame_size 224 \
  --prompt_mode classify \
  --reasoning_effort none \
  --max_completion_tokens 4096 \
  --cache_path backend/pipelines/vlm/openai/outputs/openai_test_pose_cache.jsonl \
  --resume \
  2>&1 | tee -a logs/openai-eval-test-pose.log
```

### OpenAI no-pose ablation (test)

```bash
"${ISOCOURT_PYTHON}" backend/pipelines/vlm/openai/eval_openai_stroke.py \
  --jsonl backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl \
  --split test \
  --model gpt-5.5 \
  --pose_mode none \
  --num_frames 16 \
  --frame_size 224 \
  --prompt_mode classify \
  --reasoning_effort none \
  --max_completion_tokens 4096 \
  --cache_path backend/pipelines/vlm/openai/outputs/openai_test_nopose_cache.jsonl \
  --resume \
  2>&1 | tee logs/openai-eval-test-nopose.log
```

### OpenAI in tmux (long test run)

```bash
export ISOCOURT_TMUX_SESSION=isocourt-openai-pose-test
export ISOCOURT_TMUX_REPLACE=1

tmux new-session -ds "${ISOCOURT_TMUX_SESSION}" \
  "cd ~/IsoCourt && \
   mkdir -p logs backend/pipelines/vlm/openai/outputs && \
   export ISOCOURT_VENV=\"\$HOME/miniconda3/envs/isocourt\" && \
   export ISOCOURT_PYTHON=\"\${ISOCOURT_PYTHON:-\${ISOCOURT_VENV}/bin/python}\" && \
   export OPENAI_API_KEY=\"${OPENAI_API_KEY}\" && \
   export OPENAI_VLM_MODEL=\"${OPENAI_VLM_MODEL:-gpt-5.5}\" && \
   export OPENAI_VLM_REASONING_EFFORT=\"${OPENAI_VLM_REASONING_EFFORT:-none}\" && \
   export PYTHONPATH=\${PWD}/backend:\${PYTHONPATH:-} && \
   \${ISOCOURT_PYTHON} backend/pipelines/vlm/openai/eval_openai_stroke.py \
     --jsonl backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl \
     --split test \
     --model gpt-5.5 \
     --pose_mode cache_text \
     --pose_cache_path backend/models/pose_cache_mediapipe.pt \
     --num_frames 16 \
     --frame_size 224 \
     --prompt_mode classify \
     --reasoning_effort none \
     --max_completion_tokens 4096 \
     --cache_path backend/pipelines/vlm/openai/outputs/openai_test_pose_cache.jsonl \
     --resume \
   2>&1 | tee logs/openai-eval-test-pose.log"

tmux attach -t "${ISOCOURT_TMUX_SESSION}"
```

### OpenAI no-pose in tmux (test)

```bash
export ISOCOURT_TMUX_SESSION=isocourt-openai-nopose-test
export ISOCOURT_TMUX_REPLACE=1

tmux new-session -ds "${ISOCOURT_TMUX_SESSION}" \
  "cd ~/IsoCourt && \
   mkdir -p logs backend/pipelines/vlm/openai/outputs && \
   export ISOCOURT_VENV=\"\$HOME/miniconda3/envs/isocourt\" && \
   export ISOCOURT_PYTHON=\"\${ISOCOURT_PYTHON:-\${ISOCOURT_VENV}/bin/python}\" && \
   export OPENAI_API_KEY=\"${OPENAI_API_KEY}\" && \
   export PYTHONPATH=\${PWD}/backend:\${PYTHONPATH:-} && \
   \${ISOCOURT_PYTHON} backend/pipelines/vlm/openai/eval_openai_stroke.py \
     --jsonl backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl \
     --split test \
     --model gpt-5.5 \
     --pose_mode none \
     --num_frames 16 \
     --frame_size 224 \
     --prompt_mode classify \
     --reasoning_effort none \
     --max_completion_tokens 4096 \
     --cache_path backend/pipelines/vlm/openai/outputs/openai_test_nopose_cache.jsonl \
     --resume \
   2>&1 | tee logs/openai-eval-test-nopose.log"
```

Final report should read: `stroke_type accuracy: …/N (…%) (test split)` where **N** is your sanity-check test row count.

If you get **empty responses** from gpt-5.x: keep `--reasoning_effort none`, raise `--max_completion_tokens`, or switch `--model gpt-4o`.

---

## 8. Smoke tests (1 epoch / few batches)

```bash
export ISOCOURT_TMUX_SESSION=isocourt-smoke
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh conv3d_pose --epochs 1 --max-train-batches 2
./scripts/cluster/run_train_tmux.sh jvc --epochs 1 --batch-size 2
```

---

## 9. Pull checkpoints home (laptop)

From **laptop** repo root:

```bash
./scripts/cluster/rsync_pull.sh
```

Or set `ISOCOURT_RSYNC_DEST` before training (§1) for automatic push on success.

---

## 10. `run_train_tmux.sh` model aliases

| Alias | Script |
| ----- | ------ |
| `full`, `cnn_lstm`, `resnet50` | `train_full.py` |
| `conv3d`, `conv3d_pose` | `train_conv3d.py` |
| `timesformer` | `train_timesformer.py` |
| `jvc`, `k_st_vit` | `train_jvc.py` |
| `jvc_no_xattn`, `jvc_no_cross_attn` | `train_jvc_no_xattn.py` |
| `bst_prep` | `prepare_bst_finebadminton_collated.py` (MediaPipe; legacy) |
| `bst_prep_mmpose` | `prepare_bst_finebadminton_collated_mmpose.py` |
| `bst`, `bst_baseline` | `train_bst_baseline.py` |
| `tempose`, `tempose_baseline` | `train_tempose_baseline.py` |
| `stgcn`, `stgcn_baseline` | `train_stgcn_baseline.py` |

**Test eval (no MLflow):** `scripts/cluster/run_eval_tmux.sh` → skeleton or native eval scripts with `--split test`.
| `qwen3_vl_8b`, `qwen_vl_8b` | `train_qwen3_vl_8b.py` |

---

## 11. Expected checkpoint outputs

| Model | Default on-disk name (experiment mode) |
| ----- | -------------------------------------- |
| CNN+LSTM | `backend/models/badminton_model_YYYYMMDDTHHMMSSZ.pth` |
| Conv3D | `backend/models/badminton_model_conv3d_pose_YYYYMMDDTHHMMSSZ.pth` |
| TimeSformer | `backend/models/badminton_model_timesformer_YYYYMMDDTHHMMSSZ.pth` |
| JVC | `backend/models/badminton_model_jvc_YYYYMMDDTHHMMSSZ.pth` |
| JVC no-xattn | `backend/models/badminton_model_jvc_no_xattn_YYYYMMDDTHHMMSSZ.pth` |
| BST | `backend/models/badminton_model_bst_YYYYMMDDTHHMMSSZ.pth` |
| TemPose-V | `backend/models/badminton_model_tempose_YYYYMMDDTHHMMSSZ.pth` |
| ST-GCN | `backend/models/badminton_model_stgcn_YYYYMMDDTHHMMSSZ.pth` |
| Qwen | LoRA dir under `--output_dir` (not `backend/models/*.pth`) |
| OpenAI | JSONL cache under `--cache_path` (no `.pth`) |

Logs: `logs/train-UTC.log` (or `ISOCOURT_TRAIN_LOG` if set).

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `Annotation file  not found.` (blank path) | Empty `"${LIST_FILE}"` or unset env | Use literal paths from the table at top; rerun §2/§3 commands |
| `Loading dataset from ` (blank) | Empty `"${DATA_ROOT}"` on `full` | Same; or rsync latest `train_full.py` (treats empty as default) |
| `No samples loaded` | Wrong `--data-root` / labels missing on volume | `ls backend/data/*.json`; rerun `setup_data_symlink.sh` if needed |
| tmux session already exists | Same `ISOCOURT_TMUX_SESSION` | `export ISOCOURT_TMUX_REPLACE=1` or pick a new session name |
| OpenAI empty responses | gpt-5.x reasoning ate token budget | `--reasoning_effort none --max_completion_tokens 4096` or `--model gpt-4o` |
| MMPose / mmcv install fails in `isocourt` | torch 2.10+cu128 has no mmcv wheels | Run `./scripts/cluster/install_bst_mmpose.sh`; use `ISOCOURT_PYTHON` for prep only |
| `numpy.dtype size changed` / `Expected 96, got 88` | PyPI `xtcocotools` wheel is NumPy-2-built; env has NumPy 1.x | Pin `numpy==1.26.4` + build xtcocotools from source (`--no-build-isolation --no-binary`) |
| `Failed to initialize NumPy` / still on numpy 2.2.6 with torch 2.1 | `opencv-python-headless` 5.x requires `numpy>=2` | `pip install numpy==1.26.4 opencv-python-headless==4.9.0.80` |
| `Failed to build mmcv` / `No module named mmcv` | torch2.3 index only has mmcv 2.2.0; pip tries source build | Use torch 2.1+cu121 + `pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html --only-binary mmcv` |
| `No module named 'mmcv'` after mmpose pip | mmcv never installed (build skipped) | Re-run `./scripts/cluster/install_bst_mmpose.sh`; verify `python -c "import mmcv"` before mmpose |
| `libGL.so.1: cannot open shared object file` on `import cv2` | `opencv-python` needs display libs; cluster is headless | `./scripts/cluster/ensure_opencv_headless.sh` (after `conda activate isocourt-mmpose`) |
| `chumpy` / `mmpose` pip build fails | abandoned dep; not needed for RTMPose collate | `pip install "mmpose>=1.3.0,<1.4.0" --no-deps` then `pip install json-tricks munkres pyyaml yapf rich termcolor colorama` |
