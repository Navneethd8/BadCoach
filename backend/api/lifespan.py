"""Application startup: load model, pose, detector, Gemini client."""
import json
import os
import sys

import torch
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import state

from core.model import CNN_LSTM_Model
from core.dataset import FineBadmintonDataset
from core.pose_utils import PoseEstimator
from core.badminton_detector import BadmintonPoseDetector

load_dotenv()

MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../models"))
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../models"))

MODEL_PATH: str = ""


def _pick_best_cnn_lstm_model() -> tuple[str, int]:
    """
    Read model_registry.json, pick the highest-accuracy checkpoint whose file exists,
    skipping filenames containing ``staeformer`` (different forward signature).
    Returns (abs_path, hidden_size).
    """
    registry_path = os.path.join(MODELS_DIR, "model_registry.json")
    fallback = os.path.join(MODELS_DIR, "badminton_model.pth")

    if not os.path.exists(registry_path):
        return fallback, 128

    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read model registry: {e}")
        return fallback, 128

    best_name, best_acc, best_hidden = None, -1.0, 128
    for name, meta in registry.get("models", {}).items():
        if "staeformer" in name.lower():
            continue
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            continue
        acc = meta.get("accuracy", 0.0)
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_hidden = meta.get("hidden_size", 128)

    if best_name:
        print(f"Registry: Selected {best_name} (accuracy={best_acc}%, hidden_size={best_hidden})")
        return os.path.join(MODELS_DIR, best_name), best_hidden

    return fallback, 128


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    global MODEL_PATH

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "YOUR_API_KEY_HERE":
        print("WARNING: GEMINI_API_KEY not found. Static coaching tips only.")
        state.gemini_enabled = False
        state.gemini_client = None
    else:
        try:
            state.gemini_client = genai.Client(api_key=gemini_key)
            state.gemini_enabled = True
            print(f"SUCCESS: {state.gemini_model_name} enabled via google-genai SDK.")
        except Exception as e:
            print(f"ERROR: Failed to initialize Gemini: {e}")
            state.gemini_enabled = False
            state.gemini_client = None

    dummy_root = os.path.join(os.path.dirname(__file__), "../data")
    dummy_json = os.path.join(dummy_root, "transformed_combined_rounds_output_en_evals_translated.json")

    try:
        temp_dataset = FineBadmintonDataset(dummy_root, dummy_json)
        state.dataset_metadata = temp_dataset.classes
    except Exception as e:
        print(f"Warning: Could not load dataset metadata: {e}. Using fallback.")
        state.dataset_metadata = {
            "stroke_type": ["Serve", "Clear", "Smash", "Drop", "Drive", "Net_Shot", "Lob", "Defensive_Shot", "Other"],
            "stroke_subtype": ["None", "Short_Serve", "Flick_Serve", "High_Serve", "Common_Smash", "Jump_Smash", "Full_Smash", "Stick_Smash", "Slice_Smash", "Slice_Drop", "Stop_Drop", "Reverse_Slice_Drop", "Blocked_Drop", "Flat_Lift", "High_Lift", "Net_Lift", "Attacking_Clear", "Spinning_Net", "Flat_Drive", "High_Drive", "Other"],
            "technique": ["Forehand", "Backhand", "Turnaround", "Unknown"],
            "placement": ["Straight", "Cross-court", "Body_Hit", "Over_Head", "Passing_Shot", "Wide", "Unknown"],
            "position": ["Mid_Front", "Mid_Court", "Mid_Back", "Left_Front", "Left_Mid", "Left_Back", "Right_Front", "Right_Mid", "Right_Back", "Unknown"],
            "intent": ["Intercept", "Passive", "Defensive", "To_Create_Depth", "Move_To_Net", "Early_Net_Shot", "Deception", "Hesitation", "Seamlessly", "None"],
        }

    task_classes = {k: len(v) for k, v in state.dataset_metadata.items()}
    task_classes["quality"] = 7

    MODEL_PATH, hidden_size = _pick_best_cnn_lstm_model()
    state.device = "cuda" if torch.cuda.is_available() else "cpu"

    state.model = CNN_LSTM_Model(task_classes=task_classes, hidden_size=hidden_size)

    if os.path.exists(MODEL_PATH):
        abs_path = os.path.abspath(MODEL_PATH)
        print(f"Loading model from: {abs_path}")
        try:
            state_dict = torch.load(MODEL_PATH, map_location=state.device)
            missing, unexpected = state.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"WARNING: Missing keys in checkpoint (random init): {missing}")
            if unexpected:
                print(f"WARNING: Unexpected keys in checkpoint (ignored): {unexpected}")
            print(f"SUCCESS: Model loaded from {abs_path}.")
        except Exception as e:
            print(f"ERROR: Failed to load state_dict: {e}")
            sys.exit(1)
    else:
        print(f"CRITICAL: Model file NOT found at {os.path.abspath(MODEL_PATH)}.")
        sys.exit(1)

    state.model.to(state.device)
    state.model.eval()

    state.pose_estimator = PoseEstimator()
    state.badminton_detector = BadmintonPoseDetector()
    print("Pose Estimator and Badminton Detector initialized.")

    yield
