# TimeSformer hyperparameter ranges (same split as CNN / STAEformer)

Use this for **coarse sweeps** on `val_type_acc` (what the script checkpoints on). Log everything in MLflow.

## Sweep order (high impact first)

1. **`--vit-unfreeze-last-n`**: `{0, 2, 4}` — pretrained ViT adapting to your court usually helps **val** once regularization is OK. When `n > 0`, use **`--vit-lr-mult 0.1–0.3`** (smaller than the rest of the model).
2. **`--stroke-loss-weight`**: `{1.5, 2.0, 2.5, 3.0}` — pulls the shared trunk toward stroke_type vs other heads.
3. **Scheduler**: **`--scheduler-t0`** `{10, 20, 30}`, **`--scheduler-t-mult`** `{1, 2}` — `t_mult=1` keeps restart periods stable; larger `t0` = fewer sharp LR spikes across 60 epochs.
4. **`--aug`**: `strong` (default, matches STAEformer) vs `medium` vs `mild` — if train ≫ val, try **medium/mild**.
5. **Base `--lr`**: `{5e-5, 1e-4, 2e-4}` — effective AdamW LR is **`lr × lr-mult`** (default mult **5** → `{2.5e-4, 5e-4, 1e-3}`).
6. **`--lr-mult`**: `{3, 5, 8}` — scales all non-ViT param-group LRs (ViT uses `lr × vit-lr-mult` when unfrozen).
7. **`--weight-decay`**: `{1e-2, 2e-2, 5e-2}`.
8. **`--label-smoothing`**: `{0, 0.05, 0.1}` — lower often raises **argmax** accuracy; watch val_loss.
9. **Width / depth**: **`--embed-dim`** `{128, 192}`, **`--depth`** `{4, 5, 6}` (keep `embed_dim % num_heads == 0`).
10. **`--accum-steps`**: `{2, 4, 8}` — effective batch scale without more VRAM.

## Reference table

| Flag | Default | Typical search range | Notes |
|------|---------|----------------------|--------|
| `--lr` | `1e-4` | `5e-5` – `2e-4` | × `--lr-mult` for optimizer |
| `--lr-mult` | `5` | `3` – `8` | Match spirit of STAEformer “×5 on trunk” |
| `--vit-lr-mult` | `0.25` | `0.1` – `0.5` | Only used if `--vit-unfreeze-last-n > 0` |
| `--weight-decay` | `0.01` | `0.01` – `0.05` | |
| `--label-smoothing` | `0.1` | `0` – `0.1` | |
| `--scheduler-t0` | `10` | `10` – `30` | Cosine warm restarts period (epochs) |
| `--scheduler-t-mult` | `2` | `1` or `2` | `1` = same length every restart |
| `--accum-steps` | `4` | `2` – `8` | |
| `--stroke-loss-weight` | `2.0` | `1.5` – `3.0` | Other task weights fixed in code |
| `--aug` | `strong` | `strong` / `medium` / `mild` | |
| `--vit-unfreeze-last-n` | `0` | `0` – `4` | |
| `--embed-dim` | `128` | `128`, `192` | |
| `--depth` | `4` | `4` – `6` | |
| `--num-heads` | `4` | `4`, `8` (if dim divisible) | |
| `--batch-size` | `4` | `2` – `4` (VRAM) | |

## Example commands

```bash
# Stronger stroke emphasis, milder aug, longer stable cosine period
python3 pipelines/training/train_timesformer.py \
  --stroke-loss-weight 2.5 --aug medium --scheduler-t0 20 --scheduler-t-mult 1

# Unfreeze top of ViT with small LR on those weights
python3 pipelines/training/train_timesformer.py \
  --vit-unfreeze-last-n 2 --vit-lr-mult 0.2 --weight-decay 0.02
```

## Beating CNN+LSTM / STAEformer

Same **`video_level_split`** and metrics — compare **best val_type_acc** (and optionally **val_pos_acc**) from MLflow. If train acc runs far ahead of val, favor **medium aug**, **higher weight_decay**, or **lower lr-mult** before widening the model.
