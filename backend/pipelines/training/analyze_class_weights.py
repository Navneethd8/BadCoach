"""
Temporary script: verify dataset.py class names exist in annotations and compute
class weights from actual counts. Run from backend/ or project root with:
  python backend/pipelines/training/analyze_class_weights.py

Uses the same FineBadmintonDataset loading/mapping so counts match what training sees.

Multiclass setup (how it's tackled now):
  Each task is single-label multiclass: one true class index per sample (e.g. stroke_type in [0..8]).
  Model outputs logits (B, num_classes); CrossEntropyLoss(logits, labels) with optional
  per-class weights to handle imbalance. Prediction = argmax(logits). No multi-label.
"""
import os
import sys
from collections import Counter, defaultdict

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from core.dataset import FineBadmintonDataset, get_class_weights

CAP_UNSEEN = 2.0  # weight for classes with 0 count (used in get_class_weights and below)


def main():
    data_root = os.path.join(_backend_root, "data")
    list_file = os.path.join(_backend_root, "data", "transformed_combined_rounds_output_en_evals_translated.json")

    if not os.path.exists(list_file):
        print(f"Annotation file not found: {list_file}")
        return

    print("Loading dataset (same logic as training)...")
    dataset = FineBadmintonDataset(data_root, list_file, transform=None)
    n = len(dataset.samples)
    print(f"Total samples: {n}\n")

    # --- 1. Raw annotation values (what's actually in the JSON) ---
    print("=" * 60)
    print("RAW VALUES IN ANNOTATIONS (to verify dataset.py mappings)")
    print("=" * 60)
    raw = defaultdict(set)
    for s in dataset.samples:
        raw["hit_type"].add(s.get("hit_type", "MISSING"))
        raw["ball_area"].add(s.get("ball_area", "MISSING"))
        for a in s.get("player_actions", []):
            raw["player_actions"].add(a)
        for c in s.get("shot_characteristics", []):
            raw["shot_characteristics"].add(c)
        for st in s.get("strategies", []):
            raw["strategies"].add(st)
        for sub in s.get("subtype", []):
            raw["subtype"].add(sub)
        raw["quality"].add(s.get("quality", "MISSING"))

    for key in ["hit_type", "ball_area", "player_actions", "shot_characteristics", "strategies", "subtype", "quality"]:
        vals = sorted(str(x) for x in raw[key])
        print(f"\n{key}:")
        for v in vals[:50]:
            print(f"  {repr(v)}")
        if len(vals) > 50:
            print(f"  ... and {len(vals) - 50} more")

    # --- 2. Mapped label counts (what the model sees) ---
    print("\n" + "=" * 60)
    print("MAPPED LABEL COUNTS (dataset.py class names vs actual data)")
    print("=" * 60)
    task_counts = {task: Counter() for task in dataset.classes}
    for sample in dataset.samples:
        labels = dataset._map_labels(sample)
        for task, idx in labels.items():
            task_counts[task][idx] += 1

    # Per task: list (class_name, count), mark never-seen
    never_seen = defaultdict(list)
    for task, class_list in dataset.classes.items():
        print(f"\n--- {task} ({len(class_list)} classes) ---")
        counts = task_counts[task]
        for i, name in enumerate(class_list):
            c = counts.get(i, 0)
            tag = "  [NEVER SEEN IN DATA]" if c == 0 else ""
            print(f"  {name}: {c}{tag}")
            if c == 0:
                never_seen[task].append(name)

    # --- 3. Data-driven loss weights (via get_class_weights; cap never-seen at CAP_UNSEEN) ---
    print("\n" + "=" * 60)
    print("DATA-DRIVEN LOSS WEIGHTS (CrossEntropyLoss; never-seen classes capped at", CAP_UNSEEN, ")")
    print("=" * 60)
    list_file = os.path.join(_backend_root, "data", "transformed_combined_rounds_output_en_evals_translated.json")
    for task in ["stroke_type", "position", "technique", "placement", "intent"]:
        if task not in dataset.classes:
            continue
        weights_list, name_to_weight = get_class_weights(
            data_root, list_file, task=task, cap_unseen=CAP_UNSEEN
        )
        if not weights_list:
            continue
        class_list = dataset.classes[task]
        print(f"\n# {task} (order matches dataset.classes['{task}'])")
        print(f"weights_{task} = [")
        for name, w in zip(class_list, weights_list):
            print(f"    # {name}: {w:.4f}")
        print(f"]")
        print(f"# Single line: weights_{task} = {[round(w, 4) for w in weights_list]}")
        print(f"# By name: {name_to_weight}")
        print(f"# In training use: get_class_weights(data_root, list_file, '{task}', cap_unseen={CAP_UNSEEN})")

    # --- 4. Summary of hallucinated / missing ---
    print("\n" + "=" * 60)
    print("SUMMARY: classes in dataset.py with ZERO samples in annotations")
    print("=" * 60)
    if never_seen:
        for task, names in never_seen.items():
            if names:
                print(f"  {task}: {names}")
    else:
        print("  None — every class has at least one sample.")
    print("\nDone. train_full.py and train_staeformer.py use get_class_weights() from core.dataset for stroke_type.")


if __name__ == "__main__":
    main()
