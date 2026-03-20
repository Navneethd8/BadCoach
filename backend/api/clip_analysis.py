"""
Clip analysis: sync full analysis and async streaming events (shared implementation).
"""
import asyncio
import base64
import json
import sys

import cv2
import numpy as np
import torch

from api import state
from api.config import MAX_VIDEO_DURATION_SECONDS


def run_analysis_sync(temp_file: str) -> dict:
    """Core analysis pipeline — blocking; run via asyncio.to_thread from routes."""
    cap = cv2.VideoCapture(temp_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
    print("Buffering video frames...")
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

        p_results = state.pose_estimator.process_frame(raw_frame)
        landmarks = state.pose_estimator.get_landmarks_as_list(p_results)
        if landmarks:
            pose_landmarks_list.append(landmarks[0])
        else:
            pose_landmarks_list.append([])

    cap.release()

    is_badminton_pose, pose_confidence, pose_details = state.badminton_detector.is_badminton_video(
        pose_landmarks_list, threshold=0.05
    )

    print(f"Pose Analysis: is_badminton={is_badminton_pose}, confidence={pose_confidence:.3f}")
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
                "stance_score": pose_details.get("stance_score", 0.0),
            },
        }

    timeline = []

    video_duration = total_frames / fps
    if video_duration <= 15:
        window_size_frames = max(int(fps), 1)
        step_size_frames = window_size_frames
    elif video_duration < 30:
        window_size_frames = max(int(video_duration / 15 * fps), 1)
        step_size_frames = window_size_frames
    elif video_duration < 300:
        target_window = max(video_duration / 6, 5)
        window_size_frames = int(target_window * fps)
        step_size_frames = max(window_size_frames // 2, 1)
    else:
        window_size_frames = int(60 * fps)
        step_size_frames = int(30 * fps)

    print(f"Starting sliding window analysis (FPS: {fps:.2f})...")
    sys.stdout.flush()

    for start in range(0, total_frames - window_size_frames // 2, step_size_frames):
        end = min(start + window_size_frames, total_frames)
        timestamp = f"{int(start / fps) // 60:02d}:{int(start / fps) % 60:02d}"

        indices = np.linspace(start, end - 1, 16).astype(int)
        segment_frames = [all_frames_rgb[idx] for idx in indices]
        segment_tensor = torch.from_numpy(np.array(segment_frames)).float() / 255.0
        segment_tensor = segment_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(state.device)

        middle_idx = indices[len(indices) // 2]

        ext_cap = cv2.VideoCapture(temp_file)
        ext_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
        ret, frame_for_pose = ext_cap.read()
        ext_cap.release()

        p_results = None
        annotated_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        if ret:
            p_results = state.pose_estimator.process_frame(frame_for_pose)
            annotated_frame = state.pose_estimator.draw_landmarks(frame_for_pose, p_results)

        h, w = annotated_frame.shape[:2]
        scale = 320 / max(h, w)
        pose_img = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode(".jpg", pose_img)
        pose_b64 = base64.b64encode(buffer).decode("utf-8")

        with torch.no_grad():
            outputs = state.model(segment_tensor)

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
                        7: "Advanced", 8: "Advanced+", 9: "Expert", 10: "Elite",
                    }
                    seg_results["quality_label"] = q_map.get(seg_results["quality_numeric"], "Unknown")
                elif task == "is_badminton":
                    seg_results["is_badminton_conf"] = conf
                    seg_results["is_badminton_pred"] = idx_pred
                else:
                    seg_results[task] = {
                        "label": state.dataset_metadata[task][idx_pred],
                        "confidence": conf,
                    }

            timeline.append({
                "timestamp": timestamp,
                "label": seg_results["stroke_type"]["label"],
                "confidence": seg_results["stroke_type"]["confidence"],
                "pose_image": pose_b64,
                "is_badminton_conf": seg_results.get("is_badminton_conf", 1.0),
                "is_badminton_pred": seg_results.get("is_badminton_pred", 1),
                "metrics": {
                    "subtype": seg_results.get("stroke_subtype", {"label": "None", "confidence": 0.0}),
                    "technique": seg_results.get("technique", {"label": "Unknown", "confidence": 0.0}),
                    "placement": seg_results.get("placement", {"label": "Unknown", "confidence": 0.0}),
                    "position": seg_results.get("position", {"label": "Unknown", "confidence": 0.0}),
                    "intent": seg_results.get("intent", {"label": "None", "confidence": 0.0}),
                    "quality": seg_results.get("quality_label", "Developing"),
                },
            })

    print(f"Analysis complete. Found {len(timeline)} segments.")
    sys.stdout.flush()

    print("Stage 2: Validating badminton classification from model...")
    sys.stdout.flush()

    badminton_confidences = [seg.get("is_badminton_conf", 0.0) for seg in timeline]
    badminton_predictions = [seg.get("is_badminton_pred", 0) for seg in timeline]

    avg_badminton_conf = np.mean(badminton_confidences) if badminton_confidences else 0.0
    badminton_ratio = sum(badminton_predictions) / len(badminton_predictions) if badminton_predictions else 0.0

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
                "model_passed": model_passed,
            },
        }

    print("Validation passed! Proceeding with full analysis...")
    sys.stdout.flush()

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
            "Advanced": 7, "Advanced+": 8, "Expert": 9, "Elite": 10,
        }
        numeric_quality = q_rev.get(quality_label, 5)

        pose_rating = pose_confidence * 10
        weighted_quality = (numeric_quality * 0.9) + (pose_rating * 0.1)
        numeric_quality = int(round(weighted_quality))

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

    reco_subtype = metrics.get("subtype", {}).get("label", "None") if isinstance(metrics.get("subtype"), dict) else "None"
    reco_tech = metrics.get("technique", {}).get("label", "Unknown") if isinstance(metrics.get("technique"), dict) else "Unknown"
    reco_place = metrics.get("placement", {}).get("label", "Unknown") if isinstance(metrics.get("placement"), dict) else "Unknown"
    reco_pos = metrics.get("position", {}).get("label", "Unknown") if isinstance(metrics.get("position"), dict) else "Unknown"
    reco_intent = metrics.get("intent", {}).get("label", "None") if isinstance(metrics.get("intent"), dict) else "None"

    recommendations = []
    if state.gemini_enabled and state.gemini_client:
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
                "You are a professional badminton coach. Analyze this stroke data and provide 3 concise technical tips:\n"
                f"{tactical_context}\n\n"
                "Format as single-line bullet points."
            )
            response = state.gemini_client.models.generate_content(model=state.gemini_model_name, contents=prompt)
            recommendations = [t.strip().lstrip("*-•").strip() for t in response.text.strip().split("\n") if t.strip()][:3]
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
            "position": metrics.get("position", {}),
            "intent": metrics.get("intent", {}),
        },
        "timeline": timeline,
    }


def _stream_load_and_pose_stage_sync(temp_file: str) -> dict:
    """
    Blocking OpenCV + pose + badminton gate. Runs in a thread pool so the asyncio
    loop stays responsive for SSE and other workers.
    """
    cap = cv2.VideoCapture(temp_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frame_count > 0 and fps > 0:
        estimated_duration = total_frame_count / fps
        if estimated_duration > MAX_VIDEO_DURATION_SECONDS:
            cap.release()
            return {
                "event": "error",
                "error": (
                    f"This video is {int(estimated_duration // 60)} minutes long. "
                    f"IsoCourt currently supports videos up to {MAX_VIDEO_DURATION_SECONDS // 60} minutes."
                ),
                "over_duration_limit": True,
            }

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
        return {"event": "error", "error": "Could not read video frames", "validation_failed": True}

    sample_indices = list(range(0, total_frames, max(1, total_frames // 30)))[:30]
    pose_landmarks_list = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, raw_frame = cap.read()
        if not ret:
            pose_landmarks_list.append([])
            continue
        p_results = state.pose_estimator.process_frame(raw_frame)
        landmarks = state.pose_estimator.get_landmarks_as_list(p_results)
        pose_landmarks_list.append(landmarks[0] if landmarks else [])

    cap.release()

    is_badminton_pose, pose_confidence, _pose_details = state.badminton_detector.is_badminton_video(
        pose_landmarks_list, threshold=0.05
    )

    if not is_badminton_pose and pose_confidence < 0.05:
        return {"event": "error", "error": "This does not appear to be a badminton video.", "validation_failed": True}

    return {
        "all_frames_rgb": all_frames_rgb,
        "fps": fps,
        "total_frames": total_frames,
        "pose_confidence": pose_confidence,
    }


async def run_analyze_stream_async(temp_file: str, video_hash: str):
    """
    Async streaming pipeline — yields event dicts (same shape as legacy SSE payloads).
    Final yield should be consumed as done event by caller.
    """
    pre = await asyncio.to_thread(_stream_load_and_pose_stage_sync, temp_file)
    if "event" in pre and pre["event"] == "error":
        yield pre
        return

    all_frames_rgb = pre["all_frames_rgb"]
    fps = pre["fps"]
    total_frames = pre["total_frames"]
    pose_confidence = pre["pose_confidence"]

    video_duration = total_frames / fps
    if video_duration <= 15:
        window_size_frames = max(int(fps), 1)
        step_size_frames = window_size_frames
    elif video_duration < 30:
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
        timestamp = f"{int(start / fps) // 60:02d}:{int(start / fps) % 60:02d}"

        indices = np.linspace(start, end - 1, 16).astype(int)
        segment_frames = [all_frames_rgb[i] for i in indices]
        segment_tensor = torch.from_numpy(np.array(segment_frames)).float() / 255.0
        segment_tensor = segment_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(state.device)

        middle_idx = indices[len(indices) // 2]

        ext_cap = cv2.VideoCapture(temp_file)
        ext_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
        ret, frame_for_pose = ext_cap.read()
        ext_cap.release()

        annotated_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        if ret:
            p_results = await asyncio.to_thread(state.pose_estimator.process_frame, frame_for_pose)
            annotated_frame = state.pose_estimator.draw_landmarks(frame_for_pose, p_results)

        h, w = annotated_frame.shape[:2]
        scale = 320 / max(h, w)
        pose_img = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", pose_img)
        pose_b64 = base64.b64encode(buf).decode("utf-8")

        with torch.no_grad():
            outputs = await asyncio.to_thread(state.model, segment_tensor)
            seg_results = {}
            for task, logits in outputs.items():
                probs = torch.softmax(logits, dim=1)
                idx_pred = torch.argmax(probs, dim=1).item()
                conf = probs[0, idx_pred].item()
                if task == "quality":
                    qmap = {0: 1, 1: 3, 2: 5, 3: 6, 4: 8, 5: 9, 6: 10}
                    seg_results["quality_numeric"] = qmap.get(idx_pred, 5)
                    ql = {1: "Beginner", 2: "Beginner+", 3: "Developing", 4: "Competent", 5: "Competent+", 6: "Proficient", 7: "Advanced", 8: "Advanced+", 9: "Expert", 10: "Elite"}
                    seg_results["quality_label"] = ql.get(seg_results["quality_numeric"], "Unknown")
                elif task == "is_badminton":
                    seg_results["is_badminton_conf"] = conf
                    seg_results["is_badminton_pred"] = idx_pred
                else:
                    seg_results[task] = {"label": state.dataset_metadata[task][idx_pred], "confidence": conf}

        event_data = {
            "event": "progress",
            "window": window_index,
            "timestamp": timestamp,
            "label": seg_results.get("stroke_type", {}).get("label", "Other"),
            "confidence": seg_results.get("stroke_type", {}).get("confidence", 0.0),
            "pose_image": pose_b64,
            "metrics": {
                "subtype": seg_results.get("stroke_subtype", {"label": "None", "confidence": 0.0}),
                "technique": seg_results.get("technique", {"label": "Unknown", "confidence": 0.0}),
                "placement": seg_results.get("placement", {"label": "Unknown", "confidence": 0.0}),
                "position": seg_results.get("position", {"label": "Unknown", "confidence": 0.0}),
                "intent": seg_results.get("intent", {"label": "None", "confidence": 0.0}),
                "quality": seg_results.get("quality_label", "Developing"),
            },
        }
        timeline.append(event_data)
        yield event_data
        await asyncio.sleep(0)
        window_index += 1

    clean_timeline = [{k: v for k, v in t.items() if k != "event"} for t in timeline]

    best_event = None
    valid_events = [t for t in timeline if t.get("label") != "Other"]
    if valid_events:
        best_event = max(valid_events, key=lambda x: x.get("confidence", 0))
    elif timeline:
        best_event = max(timeline, key=lambda x: x.get("confidence", 0))

    if best_event:
        action_label = best_event.get("label", "Other")
        confidence = best_event.get("confidence", 0.0)
        metrics = best_event.get("metrics", {})
        quality_label = metrics.get("quality", "Developing")
        q_rev = {
            "Beginner": 1, "Beginner+": 2, "Developing": 3,
            "Competent": 4, "Competent+": 5, "Proficient": 6,
            "Advanced": 7, "Advanced+": 8, "Expert": 9, "Elite": 10,
        }
        numeric_quality = q_rev.get(quality_label, 5)

        pose_rating = pose_confidence * 10
        weighted_quality = (numeric_quality * 0.9) + (pose_rating * 0.1)
        numeric_quality = int(round(weighted_quality))

        if confidence < 0.3:
            numeric_quality = min(numeric_quality, 4)
        elif confidence < 0.5:
            numeric_quality = min(numeric_quality, 7)
    else:
        action_label, confidence, quality_label, numeric_quality = "Other", 0.0, "Developing", 3
        metrics = {k: {"label": "Unknown", "confidence": 0.0} for k in ["subtype", "technique", "placement", "position", "intent"]}
        metrics["quality"] = "Developing"

    reco_subtype = metrics.get("subtype", {}).get("label", "None") if isinstance(metrics.get("subtype"), dict) else "None"
    reco_tech = metrics.get("technique", {}).get("label", "Unknown") if isinstance(metrics.get("technique"), dict) else "Unknown"
    reco_place = metrics.get("placement", {}).get("label", "Unknown") if isinstance(metrics.get("placement"), dict) else "Unknown"
    reco_pos = metrics.get("position", {}).get("label", "Unknown") if isinstance(metrics.get("position"), dict) else "Unknown"
    reco_intent = metrics.get("intent", {}).get("label", "None") if isinstance(metrics.get("intent"), dict) else "None"

    recommendations = []
    if state.gemini_enabled and state.gemini_client:
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
                "You are a professional badminton coach. Analyze this stroke data and provide 3 concise technical tips:\n"
                f"{tactical_context}\n\n"
                "Format as single-line bullet points."
            )
            gemini_resp = await asyncio.to_thread(
                state.gemini_client.models.generate_content,
                model=state.gemini_model_name,
                contents=prompt,
            )
            recommendations = [t.strip().lstrip("*-•").strip() for t in gemini_resp.text.strip().split("\n") if t.strip()][:3]
        except Exception:
            recommendations = ["Keep your eye on the shuttle.", "Maintain a low center of gravity.", "Prepare your racket early."]
    else:
        recommendations = ["Focus on early preparation.", "Maintain a balanced ready position."]

    summary_result = {
        "action": action_label,
        "confidence": confidence,
        "subtype": reco_subtype,
        "quality": quality_label,
        "quality_numeric": numeric_quality,
        "recommendations": recommendations,
        "tactical_analysis": {
            "technique": metrics.get("technique", {}),
            "placement": metrics.get("placement", {}),
            "position": metrics.get("position", {}),
            "intent": metrics.get("intent", {}),
        },
        "timeline": clean_timeline,
        "cache_hit": False,
    }
    state._result_cache[video_hash] = summary_result
    yield {"event": "done", "summary": summary_result}


def sse_line(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
