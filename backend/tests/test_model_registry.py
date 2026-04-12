"""model_registry v2 layout, inference selection, migration."""
import json
import os

import pytest

from core.model_registry import (
    ARCHITECTURE_CATEGORIES,
    ENV_INFERENCE_CATEGORY,
    migrate_v1_to_v2,
    normalize_registry,
    register_training_checkpoint,
    resolve_inference_category,
    resolve_inference_model_path,
    save_inference_selection,
)


def test_migrate_v1_active_becomes_primary_for_category():
    v1 = {
        "models": {
            "a.pth": {"script": "train_full.py", "accuracy": 1.0},
            "b.pth": {"script": "train_timesformer.py", "architecture": "timesformer", "accuracy": 2.0},
        },
        "active_model": "b.pth",
    }
    v2 = migrate_v1_to_v2(v1)
    ts = v2["models"]["architectures"]["timesformer"]
    assert ts["primary"]["file"] == "b.pth"
    assert ts["primary"]["experiment"] is False
    cnn = v2["models"]["architectures"]["cnn_lstm"]
    assert cnn["primary"]["file"] == "a.pth"


def test_resolve_inference_category_env_overrides_file(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    reg_path = models_dir / "model_registry.json"
    reg = migrate_v1_to_v2(
        {
            "models": {
                "c.pth": {"script": "train_full.py", "accuracy": 9.0},
            },
            "active_model": "c.pth",
        }
    )
    reg_path.write_text(json.dumps(reg), encoding="utf-8")
    save_inference_selection(str(models_dir), "timesformer")
    (models_dir / "c.pth").write_bytes(b"x")
    (models_dir / "t.pth").write_bytes(b"y")
    reg2 = json.loads(reg_path.read_text(encoding="utf-8"))
    reg2["models"]["architectures"]["timesformer"]["primary"] = {
        "file": "t.pth",
        "accuracy": 3.0,
        "script": "train_timesformer.py",
        "architecture": "timesformer",
        "experiment": False,
    }
    reg_path.write_text(json.dumps(reg2), encoding="utf-8")

    monkeypatch.setenv(ENV_INFERENCE_CATEGORY, "cnn_lstm")
    cat = resolve_inference_category(str(models_dir), json.loads(reg_path.read_text(encoding="utf-8")))
    assert cat == "cnn_lstm"


def test_register_experiment_vs_primary(tmp_path):
    models_dir = str(tmp_path / "m")
    os.makedirs(models_dir, exist_ok=True)
    register_training_checkpoint(
        models_dir,
        category="cnn_lstm",
        file_basename="p1.pth",
        meta={"accuracy": 1.0, "script": "train_full.py", "architecture": "cnn_lstm"},
        experiment=False,
    )
    register_training_checkpoint(
        models_dir,
        category="cnn_lstm",
        file_basename="e1.pth",
        meta={"accuracy": 2.0, "script": "train_full.py", "architecture": "cnn_lstm"},
        experiment=True,
    )
    with open(os.path.join(models_dir, "model_registry.json"), encoding="utf-8") as f:
        reg = normalize_registry(json.load(f))
    slot = reg["models"]["architectures"]["cnn_lstm"]
    assert slot["primary"]["file"] == "p1.pth"
    assert len(slot["registrations"]) == 1
    assert slot["registrations"][0]["file"] == "e1.pth"


def test_normalize_registry_fills_missing_architecture_slots():
    """Older JSON may omit categories added later; normalize must supply empty slots."""
    partial = {
        "schema_version": 2,
        "models": {
            "architectures": {
                "cnn_lstm": {"primary": None, "registrations": []},
            }
        },
    }
    reg = normalize_registry(partial)
    arch = reg["models"]["architectures"]
    for cat in ARCHITECTURE_CATEGORIES:
        assert cat in arch
        assert "primary" in arch[cat] and "registrations" in arch[cat]
    assert arch["conv3d_pose"]["primary"] is None


def test_resolve_inference_model_path_primary_only(tmp_path):
    models_dir = str(tmp_path / "m")
    os.makedirs(models_dir, exist_ok=True)
    reg = migrate_v1_to_v2(
        {
            "models": {
                "keep.pth": {"script": "train_full.py"},
                "ignore.pth": {"script": "train_timesformer.py", "architecture": "timesformer"},
            },
            "active_model": "keep.pth",
        }
    )
    with open(os.path.join(models_dir, "model_registry.json"), "w", encoding="utf-8") as f:
        json.dump(reg, f)
    open(os.path.join(models_dir, "keep.pth"), "w").close()
    open(os.path.join(models_dir, "ignore.pth"), "w").close()
    save_inference_selection(models_dir, "cnn_lstm")
    p = resolve_inference_model_path(models_dir, reg)
    assert p == os.path.join(models_dir, "keep.pth")
