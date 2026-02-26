from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import shutil
import os
import cv2
import torch
import numpy as np
import sys
import base64
import hashlib
import json
import time
import uuid
from google import genai
from dotenv import load_dotenv
from cachetools import TTLCache

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import CNN_LSTM_Model
from core.dataset import FineBadmintonDataset
from core.pose_utils import PoseEstimator
from core.badminton_detector import BadmintonPoseDetector

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Concurrency config — env-overridable for different deployment sizes
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "4"))
# Video duration cap (seconds). HF Spaces has 16 GB RAM; 20 min @ 30fps ≈ 5.4 GB — safe.
MAX_VIDEO_DURATION_SECONDS = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "1200"))

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
    print("WARNING: GEMINI_API_KEY not found or not set in .env. Falling back to static coaching tips.")
    gemini_enabled = False
else:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        model_name = 'gemini-3-flash-preview'
        gemini_enabled = True
        print(f"SUCCESS: {model_name} enabled via google-genai SDK.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Gemini: {e}")
        gemini_enabled = False




MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/badminton_model.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/badminton_model.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
pose_estimator = None
badminton_detector = None
dataset_metadata = None

# Concurrency primitives
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_active_jobs = 0
_active_jobs_lock = asyncio.Lock()

# Result cache: keyed by SHA-256 of file content, 1-hour TTL, up to 128 entries
_result_cache: TTLCache = TTLCache(maxsize=128, ttl=3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model and initialise pose utilities."""
    global model, pose_estimator, badminton_detector, dataset_metadata

    dummy_root = os.path.join(os.path.dirname(__file__), "../data")
    dummy_json = os.path.join(dummy_root, "transformed_combined_rounds_output_en_evals_translated.json")

    try:
        temp_dataset = FineBadmintonDataset(dummy_root, dummy_json)
        dataset_metadata = temp_dataset.classes
    except Exception as e:
        print(f"Warning: Could not load dataset metadata: {e}. Using fallback.")
        dataset_metadata = {
            "stroke_type": ["Serve", "Clear", "Smash", "Drop", "Drive", "Net_Shot", "Lob", "Defensive_Shot", "Other"],
            "stroke_subtype": ["None", "Short_Serve", "Flick_Serve", "High_Serve", "Common_Smash", "Jump_Smash", "Full_Smash", "Stick_Smash", "Slice_Smash", "Slice_Drop", "Stop_Drop", "Reverse_Slice_Drop", "Blocked_Drop", "Flat_Lift", "High_Lift", "Net_Lift", "Attacking_Clear", "Spinning_Net", "Flat_Drive", "High_Drive", "Other"],
            "technique": ["Forehand", "Backhand", "Turnaround", "Unknown"],
            "placement": ["Straight", "Cross-court", "Body_Hit", "Over_Head", "Passing_Shot", "Wide", "Unknown"],
            "position": ["Mid_Front", "Mid_Court", "Mid_Back", "Left_Front", "Left_Mid", "Left_Back", "Right_Front", "Right_Mid", "Right_Back", "Unknown"],
            "intent": ["Intercept", "Passive", "Defensive", "To_Create_Depth", "Move_To_Net", "Early_Net_Shot", "Deception", "Hesitation", "Seamlessly", "None"]
        }

    task_classes = {k: len(v) for k, v in dataset_metadata.items()}
    task_classes["quality"] = 7

    hidden_size = 128
    registry_path = os.path.join(os.path.dirname(MODEL_PATH), "model_registry.json")
    if os.path.exists(registry_path):
        try:
            with open(registry_path) as f:
                registry = json.load(f)
            m_name = os.path.basename(MODEL_PATH)
            if m_name in registry.get("models", {}):
                hidden_size = registry["models"][m_name].get("hidden_size", 128)
                acc = registry["models"][m_name].get("accuracy", "?")
                print(f"Registry: Loading {m_name} (accuracy={acc}%, hidden_size={hidden_size})")
        except Exception as e:
            print(f"Warning: Could not read model registry: {e}")

    model = CNN_LSTM_Model(task_classes=task_classes, hidden_size=hidden_size)

    if os.path.exists(MODEL_PATH):
        abs_path = os.path.abspath(MODEL_PATH)
        print(f"Attempting to load model from: {abs_path}")
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print(f"SUCCESS: Model loaded from {abs_path}.")
        except Exception as e:
            print(f"ERROR: Failed to load state_dict: {e}")
            sys.exit(1)
    else:
        print(f"CRITICAL: Model file NOT found at {os.path.abspath(MODEL_PATH)}.")
        sys.exit(1)

    model.to(device)
    model.eval()

    pose_estimator = PoseEstimator()
    badminton_detector = BadmintonPoseDetector()
    print("Pose Estimator and Badminton Detector initialized.")

    yield  # server is running
    # (shutdown logic would go here if needed)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Returns server health and current concurrency load."""
    return {
        "status": "ok",
        "active_jobs": _active_jobs,
        "max_jobs": MAX_CONCURRENT_JOBS,
        "queue_available": MAX_CONCURRENT_JOBS - _active_jobs,
    }


@app.get("/")
def read_root():
    return {"message": "Badminton Coach API is running!"}


def _run_analysis_sync(temp_file: str) -> dict:
    """
    Core analysis pipeline — runs in a thread pool via asyncio.to_thread().
    Blocking OpenCV/PyTorch calls are safe here since we're off the event loop.
    """
    cap = cv2.VideoCapture(temp_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Duration guard — reject before we buffer anything
    if total_frame_count > 0 and fps > 0:
        estimated_duration = total_frame_count / fps
        if estimated_duration > MAX_VIDEO_DURATION_SECONDS:
            cap.release()
            return {
                "error": (
                    f"This video is {int(estimated_duration // 60)} minutes long. "
                    f"IsoCourt currently supports videos up to {MAX_VIDEO_DURATION_SECONDS // 60} minutes. "
                    "For a full-game analysis, please trim to your key plays or upload quarter by quarter."
                ),
                "over_duration_limit": True,
                "validation_failed": True,
            }

    all_frames_rgb = []

    print(f"Buffering video frames...")
    sys.stdout.flush()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        all_frames_rgb.append(frame_resized)
        if len(all_frames_rgb) % 60 == 0:
            print(f" -> Buffered {len(all_frames_rgb)} frames...")
            sys.stdout.flush()
            
    total_frames = len(all_frames_rgb)
    print(f"Buffering complete. Total frames: {total_frames}")
    sys.stdout.flush()

    if total_frames == 0:
        return {"error": "Could not read video frames", "validation_failed": True}

    # Stage 1: MediaPipe Pose-Based Validation
    print("Stage 1: Analyzing poses for badminton characteristics...")
    sys.stdout.flush()

    sample_indices = list(range(0, total_frames, max(1, total_frames // 30)))[:30]
    pose_landmarks_list = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, raw_frame = cap.read()
        if not ret:
            pose_landmarks_list.append([])
            continue

        p_results = pose_estimator.process_frame(raw_frame)
        landmarks = pose_estimator.get_landmarks_as_list(p_results)
        if landmarks:
            pose_landmarks_list.append(landmarks[0])
        else:
            pose_landmarks_list.append([])

    cap.release()

    is_badminton_pose, pose_confidence, pose_details = badminton_detector.is_badminton_video(
        pose_landmarks_list, threshold=0.05
    )

    print(f"Pose Analysis: is_badminton={is_badminton_pose}, confidence={pose_confidence:.3f}")
    print(f"Details: {pose_details}")
    sys.stdout.flush()

    if not is_badminton_pose and pose_confidence < 0.05:
        return {
            "error": "This doesn't appear to be a badminton video. Please upload a video showing badminton gameplay with visible players and racket movements.",
            "validation_failed": True,
            "validation_details": {
                "stage": "pose_analysis",
                "pose_confidence": pose_confidence,
                "overhead_score": pose_details.get("overhead_score", 0.0),
                "racket_score": pose_details.get("racket_score", 0.0),
                "stance_score": pose_details.get("stance_score", 0.0)
            }
        }

    timeline = []

    # Adaptive sliding window — scales with video duration
    video_duration = total_frames / fps
    if video_duration <= 15:
        # ≤15s: 1 clip per second (e.g. 6s → 6 clips)
        window_size_frames = max(int(fps), 1)
        step_size_frames = window_size_frames
    elif video_duration < 30:
        # 15–30s: exactly 15 evenly-spaced clips
        window_size_frames = max(int(video_duration / 15 * fps), 1)
        step_size_frames = window_size_frames
    elif video_duration < 300:  # 30s – 5 min: ~6 even segments
        target_window = max(video_duration / 6, 5)  # at least 5s per window
        window_size_frames = int(target_window * fps)
        step_size_frames = max(window_size_frames // 2, 1)
    else:  # 5 min+: fixed 1-minute windows
        window_size_frames = int(60 * fps)
        step_size_frames = int(30 * fps)

    print(f"Starting sliding window analysis (FPS: {fps:.2f})...")
    sys.stdout.flush()

    for start in range(0, total_frames - window_size_frames // 2, step_size_frames):
        end = min(start + window_size_frames, total_frames)
        timestamp = f"{int(start/fps)//60:02d}:{int(start/fps)%60:02d}"

        indices = np.linspace(start, end - 1, 16).astype(int)
        segment_frames = [all_frames_rgb[idx] for idx in indices]
        segment_tensor = torch.from_numpy(np.array(segment_frames)).float() / 255.0
        segment_tensor = segment_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(device)

        middle_idx = indices[len(indices) // 2]
        
        # Open video briefly to fetch the single raw frame for Gemini pose extraction
        ext_cap = cv2.VideoCapture(temp_file)
        ext_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
        ret, frame_for_pose = ext_cap.read()
        ext_cap.release()
        
        p_results = None
        annotated_frame = np.zeros((224, 224, 3), dtype=np.uint8) # Default blank frame
        if ret:
            p_results = pose_estimator.process_frame(frame_for_pose)
            annotated_frame = pose_estimator.draw_landmarks(frame_for_pose, p_results)
        
        h, w = annotated_frame.shape[:2]
        scale = 320 / max(h, w)
        pose_img = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode('.jpg', pose_img)
        pose_b64 = base64.b64encode(buffer).decode('utf-8')

        with torch.no_grad():
            outputs = model(segment_tensor)

            seg_results = {}
            for task, logits in outputs.items():
                probs = torch.softmax(logits, dim=1)
                idx_pred = torch.argmax(probs, dim=1).item()
                conf = probs[0, idx_pred].item()

                if task == "quality":
                    quality_map_to_10 = {0: 1, 1: 3, 2: 5, 3: 6, 4: 8, 5: 9, 6: 10}
                    seg_results["quality_numeric"] = quality_map_to_10.get(idx_pred, 5)
                    q_map = {
                        1: "Beginner", 2: "Beginner+", 3: "Developing",
                        4: "Competent", 5: "Competent+", 6: "Proficient",
                        7: "Advanced", 8: "Advanced+", 9: "Expert", 10: "Elite"
                    }
                    seg_results["quality_label"] = q_map.get(seg_results["quality_numeric"], "Unknown")
                elif task == "is_badminton":
                    seg_results["is_badminton_conf"] = conf
                    seg_results["is_badminton_pred"] = idx_pred
                else:
                    seg_results[task] = {
                        "label": dataset_metadata[task][idx_pred],
                        "confidence": conf
                    }

            timeline.append({
                "timestamp": timestamp,
                "label": seg_results["stroke_type"]["label"],
                "confidence": seg_results["stroke_type"]["confidence"],
                "pose_image": pose_b64,
                "is_badminton_conf": seg_results.get("is_badminton_conf", 1.0),
                "is_badminton_pred": seg_results.get("is_badminton_pred", 1),
                "metrics": {
                    "subtype":    seg_results.get("stroke_subtype", {"label": "None", "confidence": 0.0}),
                    "technique":  seg_results.get("technique",      {"label": "Unknown", "confidence": 0.0}),
                    "placement":  seg_results.get("placement",      {"label": "Unknown", "confidence": 0.0}),
                    "position":   seg_results.get("position",       {"label": "Unknown", "confidence": 0.0}),
                    "intent":     seg_results.get("intent",         {"label": "None", "confidence": 0.0}),
                    "quality":    seg_results.get("quality_label",  "Developing"),
                }
            })

    print(f"Analysis complete. Found {len(timeline)} segments.")
    sys.stdout.flush()

    # Stage 2: Model-Based Validation
    print("Stage 2: Validating badminton classification from model...")
    sys.stdout.flush()

    badminton_confidences = [seg.get("is_badminton_conf", 0.0) for seg in timeline]
    badminton_predictions = [seg.get("is_badminton_pred", 0) for seg in timeline]

    avg_badminton_conf = np.mean(badminton_confidences) if badminton_confidences else 0.0
    badminton_ratio = sum(badminton_predictions) / len(badminton_predictions) if badminton_predictions else 0.0

    print(f"Model Classification: avg_confidence={avg_badminton_conf:.3f}, badminton_ratio={badminton_ratio:.3f}")
    sys.stdout.flush()

    pose_passed = pose_confidence >= 0.30
    model_passed = badminton_ratio >= 0.5 and avg_badminton_conf >= 0.50

    if not (pose_passed or model_passed):
        if pose_confidence < 0.20 and badminton_ratio < 0.3:
            error_msg = "This video doesn't show badminton gameplay. Please upload a video with visible badminton players, racket movements, and court action."
        elif pose_confidence < 0.30:
            error_msg = "Unable to detect badminton-specific movements (overhead strokes, lunges, racket swings). Please ensure the video clearly shows players in action."
        else:
            error_msg = "The video content doesn't match badminton gameplay patterns. Please upload a clear badminton match or practice video."

        return {
            "error": error_msg,
            "validation_failed": True,
            "validation_details": {
                "stage": "model_classification",
                "pose_confidence": pose_confidence,
                "model_confidence": avg_badminton_conf,
                "badminton_ratio": badminton_ratio,
                "pose_passed": pose_passed,
                "model_passed": model_passed
            }
        }

    print("Validation passed! Proceeding with full analysis...")
    sys.stdout.flush()

    # Pick Global Best
    best_event = None
    valid_events = [t for t in timeline if t["label"] != "Other"]
    if valid_events:
        best_event = max(valid_events, key=lambda x: x["confidence"])
    elif timeline:
        best_event = max(timeline, key=lambda x: x["confidence"])

    if best_event:
        action_label = best_event["label"]
        confidence = best_event["confidence"]
        metrics = best_event["metrics"]
        quality_label = metrics["quality"]
        q_rev = {
            "Beginner": 1, "Beginner+": 2, "Developing": 3,
            "Competent": 4, "Competent+": 5, "Proficient": 6,
            "Advanced": 7, "Advanced+": 8, "Expert": 9, "Elite": 10
        }
        numeric_quality = q_rev.get(quality_label, 5)

        pose_rating = pose_confidence * 10
        weighted_quality = (numeric_quality * 0.9) + (pose_rating * 0.1)
        numeric_quality = int(round(weighted_quality))

        if confidence < 0.3:
            numeric_quality = min(numeric_quality, 4)
        elif confidence < 0.5:
            numeric_quality = min(numeric_quality, 7)

        if confidence < 0.3:
            numeric_quality = min(numeric_quality, 4)
            if "(Poor Form/Detection)" not in quality_label:
                quality_label = f"{quality_label} (Low Confidence)"
        elif confidence < 0.5:
            numeric_quality = min(numeric_quality, 7)
    else:
        action_label, confidence, quality_label, numeric_quality = "Other", 0.0, "Developing", 3
        metrics = {k: {"label": "Unknown", "confidence": 0.0} for k in ["subtype", "technique", "placement", "position", "intent"]}
        metrics["quality"] = "Developing"

    reco_subtype = metrics.get('subtype', {}).get('label', 'None') if isinstance(metrics.get('subtype'), dict) else 'None'
    reco_tech    = metrics.get('technique', {}).get('label', 'Unknown') if isinstance(metrics.get('technique'), dict) else 'Unknown'
    reco_place   = metrics.get('placement', {}).get('label', 'Unknown') if isinstance(metrics.get('placement'), dict) else 'Unknown'
    reco_pos     = metrics.get('position', {}).get('label', 'Unknown') if isinstance(metrics.get('position'), dict) else 'Unknown'
    reco_intent  = metrics.get('intent', {}).get('label', 'None') if isinstance(metrics.get('intent'), dict) else 'None'

    recommendations = []
    if gemini_enabled:
        try:
            tactical_context = (
                f"- Stroke: {action_label} ({reco_subtype})\n"
                f"- Technique: {reco_tech}\n"
                f"- Placement: {reco_place}\n"
                f"- Court Position: {reco_pos}\n"
                f"- Tactical Intent: {reco_intent}\n"
                f"- Quality: {numeric_quality}/10 ({quality_label})"
            )
            prompt = (
                f"You are a professional badminton coach. Analyze this stroke data and provide 3 concise technical tips:\n"
                f"{tactical_context}\n\n"
                f"Format as single-line bullet points."
            )
            response = client.models.generate_content(model='gemini-3-flash-preview', contents=prompt)
            recommendations = [t.strip().lstrip('*-•').strip() for t in response.text.strip().split('\n') if t.strip()][:3]
        except Exception:
            recommendations = ["Keep your eye on the shuttle.", "Maintain a low center of gravity.", "Prepare your racket early."]
    else:
        recommendations = ["Focus on early preparation.", "Maintain a balanced ready position."]

    return {
        "action": action_label,
        "confidence": confidence,
        "subtype": reco_subtype,
        "quality": quality_label,
        "quality_numeric": numeric_quality,
        "recommendations": recommendations,
        "tactical_analysis": {
            "technique": metrics.get("technique", {}),
            "placement": metrics.get("placement", {}),
            "position":  metrics.get("position", {}),
            "intent":    metrics.get("intent", {}),
        },
        "timeline": timeline
    }


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    global _active_jobs

    # Read content for SHA-256 hash (idempotency)
    content = await file.read()
    video_hash = hashlib.sha256(content).hexdigest()

    # Return cached result if available
    if video_hash in _result_cache:
        cached = dict(_result_cache[video_hash])
        cached["cache_hit"] = True
        return cached

    # Enforce concurrency cap — non-blocking check
    if _semaphore.locked() and _semaphore._value == 0:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "We're at full capacity right now — IsoCourt is getting a lot of love! Please try again in a moment.",
                "retry_after": 30,
            }
        )

    try:
        await asyncio.wait_for(_semaphore.acquire(), timeout=0.5)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "We're at full capacity right now — IsoCourt is getting a lot of love! Please try again in a moment.",
                "retry_after": 30,
            }
        )

    async with _active_jobs_lock:
        _active_jobs += 1

    file_id = str(uuid.uuid4())
    temp_file = f"temp_{file_id}_{file.filename}"

    try:
        with open(temp_file, "wb") as buf:
            buf.write(content)

        result = await asyncio.to_thread(_run_analysis_sync, temp_file)

        # Cache successful, non-validation-error results
        if result and not result.get("validation_failed"):
            _result_cache[video_hash] = result

        result["cache_hit"] = False
        return result

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        _semaphore.release()
        async with _active_jobs_lock:
            _active_jobs -= 1


@app.post("/analyze/stream")
async def analyze_video_stream(file: UploadFile = File(...)):
    """
    SSE endpoint: streams each sliding window result as it completes.
    The client receives JSON events in real time, one per completed window.
    Final event: {"event": "done", "summary": {...}}
    """

    content = await file.read()
    video_hash = hashlib.sha256(content).hexdigest()

    # Check cache — stream the cached result instantly if available
    if video_hash in _result_cache:
        cached = _result_cache[video_hash]
        async def stream_cached():
            for i, seg in enumerate(cached.get("timeline", [])):
                event = json.dumps({"event": "progress", "window": i, **seg})
                yield f"data: {event}\n\n"
                await asyncio.sleep(0)
            summary = {k: v for k, v in cached.items() if k != "timeline"}
            summary["cache_hit"] = True
            yield f"data: {json.dumps({'event': 'done', 'summary': summary})}\n\n"
        return StreamingResponse(stream_cached(), media_type="text/event-stream")

    # Enforce concurrency cap
    if _semaphore._value <= 0:
        async def at_capacity():
            yield f"data: {json.dumps({'event': 'error', 'error': 'Server at capacity. Please try again shortly.', 'retry_after': 30})}\n\n"
        return StreamingResponse(at_capacity(), media_type="text/event-stream", status_code=503)

    file_id = str(uuid.uuid4())
    temp_file = f"temp_stream_{file_id}_{file.filename}"

    async def generate():
        global _active_jobs
        await _semaphore.acquire()
        async with _active_jobs_lock:
            _active_jobs += 1

        try:
            with open(temp_file, "wb") as buf:
                buf.write(content)

            # --- Run the streaming pipeline ---
            cap = cv2.VideoCapture(temp_file)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frame_count > 0 and fps > 0:
                estimated_duration = total_frame_count / fps
                if estimated_duration > MAX_VIDEO_DURATION_SECONDS:
                    cap.release()
                    msg = json.dumps({
                        "event": "error",
                        "error": (
                            f"This video is {int(estimated_duration // 60)} minutes long. "
                            f"IsoCourt currently supports videos up to {MAX_VIDEO_DURATION_SECONDS // 60} minutes."
                        ),
                        "over_duration_limit": True
                    })
                    yield f"data: {msg}\n\n"
                    return

            # Buffer frames (only 224x224 to save memory)
            all_frames_rgb = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames_rgb.append(cv2.resize(frame_rgb, (224, 224)))
            total_frames = len(all_frames_rgb)

            if total_frames == 0:
                cap.release()
                yield f"data: {json.dumps({'event': 'error', 'error': 'Could not read video frames', 'validation_failed': True})}\n\n"
                return

            # Stage 1: Pose Validation
            sample_indices = list(range(0, total_frames, max(1, total_frames // 30)))[:30]
            pose_landmarks_list = []

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, raw_frame = cap.read()
                if not ret:
                    pose_landmarks_list.append([])
                    continue
                p_results = pose_estimator.process_frame(raw_frame)
                landmarks = pose_estimator.get_landmarks_as_list(p_results)
                pose_landmarks_list.append(landmarks[0] if landmarks else [])

            cap.release()

            is_badminton_pose, pose_confidence, pose_details = badminton_detector.is_badminton_video(
                pose_landmarks_list, threshold=0.05
            )

            if not is_badminton_pose and pose_confidence < 0.05:
                yield f"data: {json.dumps({'event': 'error', 'error': 'This does not appear to be a badminton video.', 'validation_failed': True})}\n\n"
                return

            # Adaptive sliding window — scales with video duration
            video_duration = total_frames / fps
            if video_duration <= 15:
                # ≤15s: 1 clip per second
                window_size_frames = max(int(fps), 1)
                step_size_frames = window_size_frames
            elif video_duration < 30:
                # 15–30s: exactly 15 evenly-spaced clips
                window_size_frames = max(int(video_duration / 15 * fps), 1)
                step_size_frames = window_size_frames
            elif video_duration < 300:
                target_window = max(video_duration / 6, 5)
                window_size_frames = int(target_window * fps)
                step_size_frames = max(window_size_frames // 2, 1)
            else:
                window_size_frames = int(60 * fps)
                step_size_frames = int(30 * fps)
            window_size_frames = max(window_size_frames, 1)

            timeline = []
            window_index = 0

            for start in range(0, total_frames - window_size_frames // 2, step_size_frames):
                end = min(start + window_size_frames, total_frames)
                timestamp = f"{int(start/fps)//60:02d}:{int(start/fps)%60:02d}"

                indices = np.linspace(start, end - 1, 16).astype(int)
                segment_frames = [all_frames_rgb[i] for i in indices]
                segment_tensor = torch.from_numpy(np.array(segment_frames)).float() / 255.0
                segment_tensor = segment_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(device)

                middle_idx = indices[len(indices) // 2]
                
                ext_cap = cv2.VideoCapture(temp_file)
                ext_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
                ret, frame_for_pose = ext_cap.read()
                ext_cap.release()
                
                p_results = None
                annotated_frame = np.zeros((224, 224, 3), dtype=np.uint8) # Default blank frame
                if ret:
                    p_results = await asyncio.to_thread(pose_estimator.process_frame, frame_for_pose)
                    annotated_frame = pose_estimator.draw_landmarks(frame_for_pose, p_results)
                
                h, w = annotated_frame.shape[:2]
                scale = 320 / max(h, w)
                pose_img = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
                _, buf = cv2.imencode('.jpg', pose_img)
                pose_b64 = base64.b64encode(buf).decode('utf-8')

                with torch.no_grad():
                    outputs = await asyncio.to_thread(model, segment_tensor)
                    seg_results = {}
                    for task, logits in outputs.items():
                        probs = torch.softmax(logits, dim=1)
                        idx_pred = torch.argmax(probs, dim=1).item()
                        conf = probs[0, idx_pred].item()
                        if task == "quality":
                            qmap = {0:1,1:3,2:5,3:6,4:8,5:9,6:10}
                            seg_results["quality_numeric"] = qmap.get(idx_pred, 5)
                            ql = {1:"Beginner",2:"Beginner+",3:"Developing",4:"Competent",5:"Competent+",6:"Proficient",7:"Advanced",8:"Advanced+",9:"Expert",10:"Elite"}
                            seg_results["quality_label"] = ql.get(seg_results["quality_numeric"], "Unknown")
                        elif task == "is_badminton":
                            seg_results["is_badminton_conf"] = conf
                            seg_results["is_badminton_pred"] = idx_pred
                        else:
                            seg_results[task] = {"label": dataset_metadata[task][idx_pred], "confidence": conf}

                event_data = {
                    "event": "progress",
                    "window": window_index,
                    "timestamp": timestamp,
                    "label": seg_results.get("stroke_type", {}).get("label", "Other"),
                    "confidence": seg_results.get("stroke_type", {}).get("confidence", 0.0),
                    "pose_image": pose_b64,
                    "metrics": {
                        "subtype":   seg_results.get("stroke_subtype", {"label": "None",    "confidence": 0.0}),
                        "technique": seg_results.get("technique",      {"label": "Unknown",  "confidence": 0.0}),
                        "placement": seg_results.get("placement",      {"label": "Unknown",  "confidence": 0.0}),
                        "position":  seg_results.get("position",       {"label": "Unknown",  "confidence": 0.0}),
                        "intent":    seg_results.get("intent",         {"label": "None",     "confidence": 0.0}),
                        "quality":   seg_results.get("quality_label",  "Developing"),
                    }
                }
                timeline.append(event_data)
                yield f"data: {json.dumps(event_data)}\n\n"
                await asyncio.sleep(0)  # yield control to event loop
                window_index += 1

            # Cache and send final summary
            summary_result = {
                "action": timeline[0]["label"] if timeline else "Other",
                "confidence": max((t["confidence"] for t in timeline), default=0.0),
                "timeline": [{k: v for k, v in t.items() if k != "event"} for t in timeline],
                "cache_hit": False,
            }
            _result_cache[video_hash] = summary_result
            yield f"data: {json.dumps({'event': 'done', 'summary': summary_result})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            _semaphore.release()
            async with _active_jobs_lock:
                _active_jobs -= 1

    return StreamingResponse(generate(), media_type="text/event-stream")
