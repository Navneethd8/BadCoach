# EC2 bare-metal training (no Docker)

For GPU instances such as `g4dn.xlarge` (T4). Flow: **rsync push → bootstrap → tmux train → rsync pull**.

## Rsync quick reference

**Run these on your laptop**, not from an SSH session on EC2. From the **repo root** (the directory that contains `scripts/`). Put `EC2_HOST` and `KEY_FILE` in **`scripts/ec2.env`** (gitignored), or pass `user@host:~/IsoCourt` as the first argument.

| Direction | Command | What it syncs |
| --------- | ------- | --------------- |
| **Push** to EC2 | `./scripts/ec2/rsync_push.sh` | Whole repo → remote `~/IsoCourt/` (see script for excludes: `.venv`, `mlruns`, large data, etc.) |
| **Pull** from EC2 | `./scripts/ec2/rsync_pull.sh` | Remote **`backend/models/`** and **`mlruns/`** only → same paths locally (checkpoints, `model_registry.json`, MLflow UI data) |

Optional: `export RSYNC_EXTRA='--info=progress2'` before either script for a single progress line. If reading the PEM from an external drive fails with I/O errors, copy the key to e.g. `~/.ssh/your-key.pem` and set `KEY_FILE` to that path in `ec2.env`.

**Full-tree pull** (entire `IsoCourt` mirror, like a manual `rsync`; still from the laptop):

```bash
source scripts/ec2.env   # or set EC2_HOST / KEY_FILE by hand
rsync -avz --partial --info=progress2 \
  -e "ssh -i ${KEY_FILE}" \
  --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  "${EC2_HOST}:~/IsoCourt/" "${PWD}/"
```

Run that with `PWD` equal to your local clone root (or replace `${PWD}/` with an absolute path). Trailing slash on the remote side copies **contents** into the destination directory.

## 1. From your laptop (repo root)

Create **`scripts/ec2.env`** at the repo root (same level as `scripts/ec2/`). It is **gitignored**; put your instance SSH target and key there, for example:

```bash
EC2_HOST=ec2-user@YOUR_PUBLIC_IP
KEY_FILE=/path/to/your-key.pem
```

**NVIDIA drivers (Ubuntu on the instance):** after the first `rsync` of the repo, run:

```bash
sudo ./scripts/ec2/install_nvidia_driver_reboot.sh
```

That uses `ubuntu-drivers autoinstall`, waits 15 seconds, then reboots. Set `INSTALL_SKIP_REBOOT=1` to install only, or `INSTALL_SKIP_SLEEP=1` to reboot immediately. On **Amazon Linux** (`ec2-user`), use a GPU/DLAMI image or NVIDIA’s guide instead—this script exits with a pointer.

Then either of these works:

```bash
./scripts/ec2/rsync_push.sh
# same as: ./scripts/ec2/rsync_push.sh "${EC2_HOST}:~/IsoCourt" with ssh -i "${KEY_FILE}"
./scripts/ec2/rsync_push.sh ubuntu@OTHER:~/IsoCourt   # override host/path for one run
```

`KEY_FILE` is turned into `rsync -e "ssh -i …"` automatically when set in `ec2.env`. You can still set **`RSYNC_EXTRA`** for more ssh/rsync flags (it is appended after that).

### What `rsync_push.sh` skips (HF / prod inference only)

Not needed for stroke or VLM **pipelines** on the box; kept off to save time and disk:

- `.github/` — CI, including Hugging Face Spaces deploy workflows
- `frontend/` — web app
- `backend/api/` — FastAPI inference service (used for local/prod API, not training entrypoints)
- `backend/deploy/`, `backend/docker-compose.yml`, `backend/Dockerfile`, `backend/requirements-inference.txt` — inference / HF-style deploy wiring

`backend/pipelines/vlm/` **is included** (Qwen / VLM inference or training after stroke runs). `backend/tests/` is still included if you want to run pytest on the instance. To sync something that was excluded, run a one-off `rsync` or trim the exclude list in `rsync_push.sh`.

## 2. On EC2 (SSH in once)

```bash
cd ~/IsoCourt
chmod +x scripts/ec2/*.sh
# Default PyTorch CUDA 12.1 wheels; use TORCH_CUDA=cu124 if your driver is new enough (>= 550).
./scripts/ec2/bootstrap_ec2.sh
```

## 3. Start training (detached; safe to close SSH)

```bash
export ISOCOURT_TMUX_REPLACE=1   # optional: reuse session name
./scripts/ec2/run_train_tmux.sh timesformer --epochs 5 --batch-size 2
tmux attach -t isocourt-train    # optional: watch logs; detach with Ctrl-b d
```

Optional environment for `run_train_tmux.sh` (export **before** invoking it so the tmux pane inherits the variables):

| Variable | Effect |
| -------- | ------ |
| `ISOCOURT_RSYNC_DEST` | Non-empty SSH destination: after **success**, rsync `backend/models/` and `mlruns/` there (e.g. `you@home:~/IsoCourt/`). |
| `ISOCOURT_SHUTDOWN` | If `1`, after **success** run `sudo shutdown -h now` (stop the instance to save cost). |
| `ISOCOURT_SHUTDOWN_ON_ERROR` | If `1`, when training exits **non-zero** run `sudo shutdown -h now` (halt on failure; does not run success rsync/shutdown-for-success logic). |

`ISOCOURT_RSYNC_EXTRA` can hold extra rsync arguments (one string). Other knobs (`ISOCOURT_VENV`, `ISOCOURT_TMUX_SESSION`, `ISOCOURT_TMUX_REPLACE`, `ISOCOURT_TRAIN_LOG`) are documented in the header of `scripts/ec2/run_train_tmux.sh`.

Example:

```bash
export ISOCOURT_RSYNC_DEST="you@home:~/IsoCourt/"
export ISOCOURT_SHUTDOWN=1
# export ISOCOURT_SHUTDOWN_ON_ERROR=1   # optional: halt instance when training fails
./scripts/ec2/run_train_tmux.sh timesformer
```

## 4. Pull checkpoints home

Same as the **Pull** row in [Rsync quick reference](#rsync-quick-reference) above.

```bash
./scripts/ec2/rsync_pull.sh
# or: ./scripts/ec2/rsync_pull.sh ubuntu@YOUR_HOST:~/IsoCourt
```

## Model names for `run_train_tmux.sh`

| Argument      | Script                   |
| ------------- | ------------------------ |
| `cnn_lstm`    | `train_full.py`          |
| `conv3d`      | `train_conv3d.py`        |
| `timesformer` | `train_timesformer.py`   |
| `st_tr`       | `train_st_tr.py` (legacy dual-transformer) |
| `gcn_st_tr`   | `train_gcn_st_tr.py` — upstream ST-TR (`build_official_st_tr`); uses `backend/models/pose_cache_mediapipe.pt` |
| `st_tr_vit`   | `train_st_tr_vit.py` — ST-TR + ViT fusion; same pose cache |
| `bst_prep`    | `prepare_bst_finebadminton_collated.py` |
| `bst_baseline`| `train_bst_baseline.py`  |

Extra CLI flags are passed through to the trainer (see each script’s `--help`).

### BST baseline (two-step)

The BST baseline requires **two** separate runs — pose extraction then training:

```bash
# Step 1: build collated .npy tensors (CPU-bound, runs MediaPipe on every clip)
./scripts/ec2/run_train_tmux.sh bst_prep \
  --data-root backend/data \
  --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \
  --output-dir backend/data/bst_finebadminton_collated \
  --sequence-length 30 --pose-style JnB_bone

# Step 2 (after step 1 finishes): train BST model (GPU)
export ISOCOURT_TMUX_REPLACE=1
./scripts/ec2/run_train_tmux.sh bst_baseline \
  --collated-root backend/data/bst_finebadminton_collated \
  --sequence-length 30 --pose-style JnB_bone \
  --model-name BST_CG_AP --epochs 80 --batch-size 64
```

## Troubleshooting

- `torch.cuda.is_available()` false after bootstrap → wrong PyTorch build; reinstall with the `cu121` / `cu124` index from `bootstrap_ec2.sh`.
- Driver error vs CUDA runtime → upgrade NVIDIA driver or lower `TORCH_CUDA`.
- `import cv2` / `mediapipe` fails → rerun bootstrap without `SKIP_APT=1` so GL/EGL packages install.
