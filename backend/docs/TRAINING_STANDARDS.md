# IsoCourt training standards (comparable runs)

Primary trainers should match these settings so benchmarks are fair. Constants live in `core/training_standards.py`.

## Clip & pose (all comparable models)

| Setting | Value | Notes |
|--------|-------|--------|
| `sequence_length` | **16** | `np.linspace` over clip window in `FineBadmintonDataset` |
| `frame_interval` | **2** | Dataset default |
| Pose cache | `(N, 16, 33, 3)` | `models/pose_cache_mediapipe.pt` |
| Joints | MediaPipe **33** | x, y, z |

## Split & sampling

| Setting | Value |
|--------|-------|
| Split | `video_level_split` — video-level **70% / 10% / 20%** (seed 42) |
| Split seed | **42** (`core/split.py`) |
| Train sampler | `WeightedRandomSampler` on `stroke_type` (pose/video multitask trainers) |

## Optimization defaults

| Setting | Value |
|--------|-------|
| Batch size | **4** (`DEFAULT_TRAIN_BATCH_SIZE`) |
| Epochs | **60** |
| LR | **5e-4** |
| Grad accumulation | **4** |
| Grad clip | **1.0** |
| Multitask loss weights | stroke_type=2, position=1, technique/placement/intent/quality=0.5 |

## MLflow

| Setting | Value |
|--------|-------|
| Tracking URI | `file:<backend>/mlruns` (unless `MLFLOW_TRACKING_URI` set) |
| Log params | Use `common_mlflow_clip_params()` + model-specific fields |

## Primary trainers (should follow table)

- `train_full.py` (CNN-LSTM)
- `train_conv3d.py`
- `train_timesformer.py`
- `train_jvc.py` / `train_jvc_no_xattn.py`

## External baselines (different by design)

- `train_bst_baseline.py` / `train_tempose_baseline.py` / `train_stgcn_baseline.py` — collated tensors, batch 64
- Each writes to its own registry category: `bst`, `tempose`, `stgcn` (default checkpoints `badminton_model_{category}.pth`; catalog only; not API inference)

## Train (tmux)

Use the shared EC2/local launcher (same as other models):

```bash
./scripts/ec2/run_train_tmux.sh jvc --epochs 60 --batch-size 4
```

## Verify

```bash
python backend/scripts/verify_training_standards.py
```
