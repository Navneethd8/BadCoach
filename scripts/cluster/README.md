# Cluster GPU training (nd17)

For the UW AI Clinic node **`nd17@is-aiclinic.ischool.uw.edu`**. Flow: **rsync code → rsync data → symlink → bootstrap → tmux train → rsync pull**.

Code lives under **`~/IsoCourt/`** in your home. The FineBadminton-20K dataset goes on the **shared cluster data volume** (not `~/data/...`) and is wired in via a symlink at `~/IsoCourt/backend/data`.

For the legacy EC2 workflow, see **[`scripts/ec2/README.md`](../ec2/README.md)**.

## Rsync quick reference

**Run these on your laptop**, from the **repo root**. Copy **`scripts/cluster.env.example`** to **`scripts/cluster.env`** (gitignored):

```bash
CLUSTER_HOST=nd17@is-aiclinic.ischool.uw.edu
KEY_FILE=                        # leave empty to type NetID password at prompt
REMOTE_REPO='~/IsoCourt'         # code in your home (quote ~)
REMOTE_DATA=/data/models/navneeth       # shared volume — edit to your lab's mount
```

### Find `REMOTE_DATA` on nd17

SSH in and locate the shared mount (path varies by lab):

```bash
ssh nd17@is-aiclinic.ischool.uw.edu
df -h
ls -ld /data /mnt/data /shared /scratch 2>/dev/null
```

Set `REMOTE_DATA` in `scripts/cluster.env` to a directory you can write on that volume, e.g. `/data/models/navneeth` or `/mnt/data/yourgroup/isocourt`. Create it if needed:

```bash
mkdir -p /data/models/navneeth    # use the path your admins expect
```

| Direction | Command | What it syncs |
| --------- | ------- | --------------- |
| **Push code** | `./scripts/cluster/rsync_push_code.sh` | Repo → remote `~/IsoCourt/` (excludes `backend/data/`) |
| **Push VLM only** | `./scripts/cluster/rsync_push_vlm.sh` | Just 16-frame VLM scaffold (fast; no full repo sync) |
| **Push data** | `./scripts/cluster/rsync_push_data.sh` | `FineBadminton-20K/` + labels JSON → **`REMOTE_DATA`** on shared volume |
| **Pull** | `./scripts/cluster/rsync_pull.sh` | Remote `backend/models/` + `backend/mlruns/` → local |

Optional progress on large transfers:

```bash
# macOS (built-in rsync): per-file progress
export RSYNC_EXTRA='--progress'

# Linux / Homebrew GNU rsync (brew install rsync): single summary line
export RSYNC_EXTRA='--info=progress2'
```

Override host/path for one run:

```bash
./scripts/cluster/rsync_push_code.sh nd17@is-aiclinic.ischool.uw.edu:~/IsoCourt
./scripts/cluster/rsync_push_data.sh nd17@is-aiclinic.ischool.uw.edu:/data/models/navneeth
```

## 1. From your laptop (repo root)

```bash
cp scripts/cluster.env.example scripts/cluster.env
# edit CLUSTER_HOST / REMOTE_DATA if needed

# Full code sync (slow):
# ./scripts/cluster/rsync_push_code.sh

# VLM scaffold only (recommended before Qwen train):
./scripts/cluster/rsync_push_vlm.sh

export RSYNC_EXTRA='--progress'   # macOS; use --info=progress2 on GNU rsync
./scripts/cluster/rsync_push_data.sh
```

If the dataset is **not** on your laptop, skip `rsync_push_data.sh` and prepare on-cluster after the symlink step (see below).

### What `rsync_push_code.sh` skips

Same as EC2 (no frontend, api, deploy, tests, etc.) **plus** the entire **`backend/data/`** tree so code sync never overwrites the data volume.

## 2. On nd17 (SSH in once)

```bash
ssh nd17@is-aiclinic.ischool.uw.edu
cd ~/IsoCourt
chmod +x scripts/cluster/*.sh

# Wire ~/data/models/navneeth → backend/data
./scripts/cluster/setup_data_symlink.sh
ls -la backend/data/FineBadminton-20K/videos/

# If dataset was not rsync'd from laptop:
# source .venv/bin/activate   # after bootstrap
# python backend/pipelines/vlm/common/prepare_finebadminton_20k.py
# python backend/pipelines/vlm/common/prepare_finebadminton_20k.py --skip-download --extract-training-frames

./scripts/cluster/bootstrap_cluster.sh
```

`bootstrap_cluster.sh` assumes NVIDIA drivers are already installed (`nvidia-smi` must work). Use `TORCH_CUDA=cu124` if driver ≥ 550, else default `cu121`.

## 3. Start training (detached; safe to close SSH)

```bash
export ISOCOURT_TMUX_REPLACE=1
./scripts/cluster/run_train_tmux.sh timesformer --epochs 5 --batch-size 2
tmux attach -t isocourt-train    # detach with Ctrl-b d
```

### Qwen3-VL-8B (16-frame + pose cache)

```bash
chmod +x scripts/cluster/prepare_vlm_16frame.sh
./scripts/cluster/prepare_vlm_16frame.sh

# Install Unsloth stack once (GPU node):
# pip install -r backend/pipelines/vlm/common/requirements-unsloth-vlm.txt

./scripts/cluster/run_train_tmux.sh qwen3_vl_8b --jsonl backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl \
  --pose_mode cache_text \
  --num_frames 16 --frame_size 224 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --output_dir backend/pipelines/vlm/qwen-8b/outputs/qwen3_vl_8b_16frame_lora
```

Ensure `backend/models/pose_cache_span_linspace.pt` exists before training with `--pose_mode cache_text` (build via `build_full_pose_cache.py --sampling span_linspace`).

Optional: after success, push checkpoints to your laptop (set before invoking `run_train_tmux.sh`):

```bash
export ISOCOURT_RSYNC_DEST="you@laptop:~/IsoCourt/"
```

Unlike EC2, there is **no** `ISOCOURT_SHUTDOWN` — this is a shared persistent host.

## 4. Pull checkpoints home

```bash
./scripts/cluster/rsync_pull.sh
```

## Remote layout

```
~/IsoCourt/                         # code, venv, models, mlruns (your home)
├── backend/
│   ├── data → /data/models/navneeth       # symlink to shared volume
│   ├── models/
│   └── pipelines/
└── scripts/cluster/

/data/models/navneeth/                     # shared cluster data volume (REMOTE_DATA)
├── FineBadminton-20K/
│   ├── videos/
│   └── dataset/image/
└── transformed_combined_rounds_output_en_evals_translated.json
```

(`/data/models/navneeth` is an example — use your lab's actual mount path in `REMOTE_DATA`.)

## Model names for `run_train_tmux.sh`

Same aliases as EC2 — see **[`scripts/ec2/README.md`](../ec2/README.md#model-names-for-run_train_tmuxsh)**.

## Troubleshooting

- **`backend/data` is a real directory, not a symlink:** re-run `setup_data_symlink.sh` (it backs up an existing dir to `backend/data.bak.*`).
- **`mkdir ... failed: No such file or directory`:** `REMOTE_DATA` is wrong or the parent mount does not exist. Run `df -h` on nd17 and set `REMOTE_DATA` to a writable path on the shared volume, not `~/data/...`.
- **Slow training without JPEGs:** run `prepare_finebadminton_20k.py --skip-download --extract-training-frames` on-cluster.
- **SSH auth:** if `KEY_FILE` is empty, ensure `ssh nd17@is-aiclinic.ischool.uw.edu` works via your default key or agent.
