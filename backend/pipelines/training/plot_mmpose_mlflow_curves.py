"""
Export training curves (loss / accuracy) from MLflow for MMPose skeleton baselines.

Uses runs tagged ``collated_data=mmpose`` or whose logged ``collated_root`` path
contains ``mmpose`` (covers older runs if the path was logged manually).

Default experiments: BST, ST-GCN, TemPose baselines.

Example::

  cd backend
  python pipelines/training/plot_mmpose_mlflow_curves.py \\
    --tracking-uri file:$(pwd)/mlruns \\
    --out-dir data/figures/mmpose_training

  # Custom MLflow filter (see MLflow search_runs filter_string docs)
  python pipelines/training/plot_mmpose_mlflow_curves.py --filter "tags.user = 'foo'"
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

import mlflow
from mlflow.tracking import MlflowClient
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_EXPERIMENTS = (
    "IsoCourt_Training_BST_Baseline",
    "IsoCourt_Training_STGCN_Baseline",
    "IsoCourt_Training_TemPose_Baseline",
)

MMPOSE_FILTER = (
    "tags.collated_data = 'mmpose' OR params.collated_root LIKE '%mmpose%'"
)


def _default_tracking_uri() -> str:
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    return f"file:{os.path.join(_backend_root, 'mlruns')}"


def _default_out_dir() -> str:
    return os.path.join(_backend_root, "data", "figures", "mmpose_training")


def _safe_filename(s: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", s).strip("_")
    return s or "run"


def _metric_series(client: MlflowClient, run_id: str, key: str):
    hist = client.get_metric_history(run_id, key)
    if not hist:
        return [], []
    hist = sorted(hist, key=lambda m: m.step)
    return [m.step for m in hist], [m.value for m in hist]


def _format_run_title(run) -> str:
    ts = run.info.start_time
    if ts:
        dt = _dt.datetime.utcfromtimestamp(ts / 1000.0)
        tstr = dt.strftime("%Y-%m-%d %H:%M UTC")
    else:
        tstr = "?"
    name = run.data.tags.get("mlflow.runName") or ""
    if name:
        return f"{name}  |  {run.info.run_id[:8]}…  |  {tstr}"
    return f"{run.info.run_id[:8]}…  |  {tstr}"


def plot_single_run(client: MlflowClient, run, exp_name: str, out_path: str) -> None:
    run_id = run.info.run_id
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, keys, ylabel in (
        (axes[0], (("train_loss", "Train"), ("val_loss", "Val")), "Loss"),
        (axes[1], (("train_acc", "Train"), ("val_acc", "Val")), "Accuracy (%)"),
    ):
        for key, label in keys:
            steps, vals = _metric_series(client, run_id, key)
            if steps:
                ax.plot(steps, vals, label=label, marker="o", markersize=2, linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{exp_name}\n{_format_run_title(run)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _peak_val_acc(client: MlflowClient, run_id: str) -> float:
    _, vals = _metric_series(client, run_id, "val_acc")
    return max(vals) if vals else float("-inf")


def plot_val_acc_summary(
    client: MlflowClient,
    runs_df,
    exp_id_to_name: dict[str, str],
    out_path: str,
) -> None:
    if runs_df is None or runs_df.empty:
        return

    best_by_exp: dict[str, tuple[str, float]] = {}
    for _, row in runs_df.iterrows():
        eid = str(row["experiment_id"])
        rid = row["run_id"]
        peak = _peak_val_acc(client, rid)
        prev = best_by_exp.get(eid)
        if prev is None or peak > prev[1]:
            best_by_exp[eid] = (rid, peak)

    fig, ax = plt.subplots(figsize=(8, 5))
    props = {
        "BST": {"color": "#1f77b4", "ls": "-"},
        "ST-GCN": {"color": "#ff7f0e", "ls": "--"},
        "TemPose": {"color": "#2ca02c", "ls": "-."},
    }

    for eid, (rid, peak) in best_by_exp.items():
        label_base = exp_id_to_name.get(eid, eid)
        short = (
            "BST" if "BST" in label_base and "TemPose" not in label_base
            else "ST-GCN" if "STGCN" in label_base or "ST-GCN" in label_base
            else "TemPose" if "TemPose" in label_base
            else label_base
        )
        sty = props.get(short, {"color": None, "ls": "-"})
        steps, vals = _metric_series(client, rid, "val_acc")
        if not steps:
            continue
        ax.plot(
            steps,
            vals,
            label=f"{short} (peak {peak:.1f}%)",
            linewidth=2.0,
            color=sty["color"],
            linestyle=sty["ls"],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("MMPose skeleton baselines — validation accuracy (best run per model)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot MLflow training curves for MMPose baseline runs.")
    p.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI or file:<backend>/mlruns)",
    )
    p.add_argument("--out-dir", default=None, help="Directory for PNG files")
    p.add_argument(
        "--experiments",
        default=",".join(DEFAULT_EXPERIMENTS),
        help="Comma-separated MLflow experiment names",
    )
    p.add_argument(
        "--filter",
        default=MMPOSE_FILTER,
        help="MLflow search_runs filter_string (default: MMPose runs)",
    )
    p.add_argument("--no-per-run", action="store_true", help="Skip individual run PNGs")
    p.add_argument("--no-summary", action="store_true", help="Skip combined val-acc figure")
    args = p.parse_args()

    uri = args.tracking_uri or _default_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    out_dir = os.path.abspath(args.out_dir or _default_out_dir())
    os.makedirs(out_dir, exist_ok=True)

    exp_names = [x.strip() for x in args.experiments.split(",") if x.strip()]
    exp_ids: list[str] = []
    exp_id_to_name: dict[str, str] = {}
    for name in exp_names:
        exp = client.get_experiment_by_name(name)
        if exp is None:
            print(f"[WARN] Experiment not found: {name}")
            continue
        exp_ids.append(exp.experiment_id)
        exp_id_to_name[exp.experiment_id] = name

    if not exp_ids:
        print("No experiments found. Train a baseline or set --experiments / --tracking-uri.")
        sys.exit(1)

    runs_df = mlflow.search_runs(
        experiment_ids=exp_ids,
        filter_string=args.filter or "",
        output_format="pandas",
    )

    if runs_df.empty:
        print(
            "No runs matched the filter. Try:\n"
            "  - Point --tracking-uri at the directory that contains mlruns/ from training.\n"
            "  - Use a looser --filter, or re-run training so collated_root / collated_data are logged."
        )
        sys.exit(2)

    print(f"Matched {len(runs_df)} run(s). Writing figures to {out_dir}")

    if not args.no_per_run:
        for _, row in runs_df.iterrows():
            run = client.get_run(row["run_id"])
            eid = str(run.info.experiment_id)
            exp_name = exp_id_to_name.get(eid, eid)
            fname = f"{_safe_filename(exp_name)}_{row['run_id'][:8]}.png"
            plot_single_run(client, run, exp_name, os.path.join(out_dir, fname))
            print(f"  wrote {fname}")

    if not args.no_summary:
        summary_path = os.path.join(out_dir, "mmpose_val_acc_summary.png")
        plot_val_acc_summary(client, runs_df, exp_id_to_name, summary_path)
        print(f"  wrote mmpose_val_acc_summary.png")

    print("Done.")


if __name__ == "__main__":
    main()
