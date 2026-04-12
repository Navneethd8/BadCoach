"""Application startup: load model, pose, detector, Gemini client."""
import os
import sys

import torch
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import state

from core.dataset import FineBadmintonDataset
from core.pose_utils import PoseEstimator
from core.badminton_detector import BadmintonPoseDetector
from api.model_loader import load_registry, load_stroke_model, resolve_model_path
from core.finebadminton_dataset_spec import resolve_inference_dataset_layout
from core.model_registry import registry_meta_for_checkpoint

load_dotenv()

MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../models"))
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../models"))

MODEL_PATH: str = ""


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

    registry = load_registry(MODELS_DIR)
    path_preview = resolve_model_path(MODELS_DIR, registry)
    reg_meta: dict = {}
    if path_preview:
        reg_meta = registry_meta_for_checkpoint(registry, path_preview)
    layout = resolve_inference_dataset_layout(MODELS_DIR, reg_meta)
    state.inference_dataset_id = layout["dataset_id"]
    state.inference_dataset_source = str(layout.get("source") or "")
    state.inference_frames_policy = str(layout.get("frames") or "")
    state.inference_data_root = layout["data_root"]
    state.inference_list_file = layout["list_file"]
    print(
        f"Inference dataset: {state.inference_dataset_id} "
        f"(frames={state.inference_frames_policy})\n"
        f"  data_root={state.inference_data_root}\n"
        f"  list_file={state.inference_list_file}"
    )

    try:
        temp_dataset = FineBadmintonDataset(state.inference_data_root, state.inference_list_file)
        state.dataset_metadata = temp_dataset.classes
    except Exception as e:
        print(f"Warning: Could not load dataset metadata from 20K paths: {e}. Using fallback.")
        state.dataset_metadata = {
            "stroke_type": ["Serve", "Clear", "Smash", "Drop", "Drive", "Net_Shot", "Lob", "Defensive_Shot", "Other"],
            "stroke_subtype": ["None", "Short_Serve", "Flick_Serve", "High_Serve", "Common_Smash", "Jump_Smash", "Full_Smash", "Stick_Smash", "Slice_Smash", "Slice_Drop", "Stop_Drop", "Reverse_Slice_Drop", "Blocked_Drop", "Flat_Lift", "High_Lift", "Attacking_Clear", "Spinning_Net", "Flat_Drive", "High_Drive", "High_Block", "Continuous_Net_Kills"],
            "technique": ["Forehand", "Backhand", "Unknown"],
            "placement": ["Straight", "Cross-court", "Body_Hit", "Over_Head", "Passing_Shot", "Wide", "Net_Fault", "Out", "Repeat", "Unknown"],
            "position": ["Mid_Front", "Mid_Court", "Mid_Back", "Left_Front", "Left_Mid", "Left_Back", "Right_Front", "Right_Mid", "Right_Back", "Unknown"],
            "intent": ["Intercept", "Passive", "Defensive", "To_Create_Depth", "Move_To_Net", "Early_Net_Shot", "Deception", "Hesitation", "Seamlessly", "None"],
        }

    task_classes = {k: len(v) for k, v in state.dataset_metadata.items()}
    task_classes["quality"] = 7

    state.device = "cuda" if torch.cuda.is_available() else "cpu"

    path = resolve_model_path(MODELS_DIR, registry)
    if not path:
        print(f"CRITICAL: No model checkpoint found under {MODELS_DIR}.")
        sys.exit(1)

    MODEL_PATH = path
    abs_path = os.path.abspath(MODEL_PATH)
    print(f"Loading model from: {abs_path}")

    try:
        state.model, arch = load_stroke_model(MODEL_PATH, task_classes, registry, state.device)
        state.model_architecture = arch
        print(f"Architecture: {arch}")
    except Exception as e:
        print(f"ERROR: Failed to build/load model: {e}")
        sys.exit(1)

    state.model.to(state.device)
    state.model.eval()

    state.pose_estimator = PoseEstimator()
    state.badminton_detector = BadmintonPoseDetector()
    print("Pose Estimator and Badminton Detector initialized.")

    yield
