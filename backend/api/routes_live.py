"""Live session: capacity gate + WebSocket frame-by-frame analysis."""
from __future__ import annotations

import asyncio
import base64
import json
import uuid

import cv2
import numpy as np
import torch
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from api import state

router = APIRouter(tags=["live"])


@router.post("/sessions")
async def start_live_session():
    """
    Reserve a live coaching slot. Returns 503 when MAX_CONCURRENT_LIVE_SESSIONS is reached.
    """
    if state._live_semaphore._value <= 0:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "IsoCourt cannot start a live session right now — capacity limits reached.",
                "retry_after": 120,
            },
        )
    await state._live_semaphore.acquire()
    session_id = str(uuid.uuid4())
    async with state._live_lock:
        state._live_sessions[session_id] = None
    return {"session_id": session_id, "status": "ready"}


@router.delete("/sessions/{session_id}")
async def end_live_session(session_id: str):
    async with state._live_lock:
        if session_id in state._live_sessions:
            del state._live_sessions[session_id]
            state._live_semaphore.release()
    return {"ok": True}


def _decode_frame(raw: bytes) -> np.ndarray | None:
    """Decode a JPEG/PNG blob into a BGR numpy array."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


STROKE_CONFIDENCE_FLOOR = 0.25
BADMINTON_PROB_MIN = 0.55
POSE_CONF_MIN = 0.05
RING_SIZE = 16
POSE_SAMPLE_COUNT = 4
POSE_CHECK_INTERVAL = 10
INFERENCE_INTERVAL_S = 1.0


def _run_window_inference(frames_rgb: list[np.ndarray]) -> dict:
    """Run the sliding-window model on exactly 16 (224x224) RGB frames. Blocking."""
    segment = np.array(frames_rgb)  # (16, 224, 224, 3)
    tensor = torch.from_numpy(segment).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0).to(state.device)  # (1, 16, 3, 224, 224)

    with torch.no_grad():
        outputs = state.model(tensor)

    seg_results: dict = {}
    for task, logits in outputs.items():
        probs = torch.softmax(logits, dim=1)
        idx_pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, idx_pred].item()

        if task == "quality":
            qmap = {0: 1, 1: 3, 2: 5, 3: 6, 4: 8, 5: 9, 6: 10}
            seg_results["quality_numeric"] = qmap.get(idx_pred, 5)
            ql = {1: "Beginner", 2: "Beginner+", 3: "Developing", 4: "Competent",
                  5: "Competent+", 6: "Proficient", 7: "Advanced", 8: "Advanced+",
                  9: "Expert", 10: "Elite"}
            seg_results["quality_label"] = ql.get(seg_results["quality_numeric"], "Unknown")
        elif task == "is_badminton":
            seg_results["badminton_prob"] = probs[0, 1].item()
        else:
            seg_results[task] = {
                "label": state.dataset_metadata[task][idx_pred],
                "confidence": conf,
            }

    stroke = seg_results.get("stroke_type", {"label": "Other", "confidence": 0.0})
    if stroke["confidence"] < STROKE_CONFIDENCE_FLOOR:
        stroke = {"label": "Other", "confidence": stroke["confidence"]}

    return {
        "label": stroke["label"],
        "confidence": stroke["confidence"],
        "badminton_prob": seg_results.get("badminton_prob", 1.0),
        "metrics": {
            "subtype": seg_results.get("stroke_subtype", {"label": "None", "confidence": 0.0}),
            "technique": seg_results.get("technique", {"label": "Unknown", "confidence": 0.0}),
            "placement": seg_results.get("placement", {"label": "Unknown", "confidence": 0.0}),
            "position": seg_results.get("position", {"label": "Unknown", "confidence": 0.0}),
            "intent": seg_results.get("intent", {"label": "None", "confidence": 0.0}),
            "quality": seg_results.get("quality_label", "Developing"),
        },
    }


def _run_pose_check(frames_rgb: list[np.ndarray]) -> float:
    """Pose-based badminton gate on a sample of frames. Blocking."""
    landmarks_list = []
    for frame_rgb in frames_rgb:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        p_results = state.pose_estimator.process_frame(frame_bgr)
        landmarks = state.pose_estimator.get_landmarks_as_list(p_results)
        landmarks_list.append(landmarks[0] if landmarks else [])
    _, pose_conf, _ = state.badminton_detector.is_badminton_video(landmarks_list, threshold=0.0)
    return pose_conf

import time

_commentary_cache: dict[str, str] = {}
_COMMENTARY_CACHE_MAX = 128
_gemini_backoff_until: float = 0.0
_GEMINI_BACKOFF_SECONDS = 30

_FALLBACK_TIPS: dict[str, list[str]] = {
    "Serve": [
        "Keep your elbow high and snap your wrist at contact for a tighter serve.",
        "Aim deep to the back tramline — push them onto the back foot early.",
        "Vary between short and flick serves to keep your opponent guessing.",
    ],
    "Smash": [
        "Rotate your torso and transfer weight forward for more power on your smash.",
        "Aim your smash to the body or sidelines — avoid the centre where they can block easily.",
        "Get under the shuttle early and contact it at the highest point you can reach.",
    ],
    "Clear": [
        "Use a full arm swing and follow through high to get depth on your clears.",
        "Push clears all the way to the baseline — a short clear is an invitation to attack.",
        "Recover to centre court quickly after playing your clear.",
    ],
    "Drop": [
        "Disguise your drop with the same arm motion as a clear to deceive your opponent.",
        "Aim just over the net with a soft touch — the steeper the angle, the harder the reply.",
        "Follow your drop to the net to cut off the return early.",
    ],
    "Net_Shot": [
        "Keep a relaxed grip and gentle touch — let the shuttle tumble over the net.",
        "Try to take net shots as early and as high as possible above the tape.",
        "Stay balanced on the balls of your feet so you can push back quickly.",
    ],
    "Drive": [
        "Keep the shuttle flat and fast — use a short punchy swing from the elbow.",
        "Aim cross-court drives to open up the sideline for your next shot.",
        "Stay compact with your swing — big backswings lose you time in drive exchanges.",
    ],
    "Lob": [
        "Play high lobs deep to the baseline to buy recovery time.",
        "Add variety — mix attacking low lobs with high defensive ones.",
        "Push off from a low lunge to get back to centre after your lob.",
    ],
    "Defensive_Shot": [
        "Keep your racket up in front of you and react with short compact swings.",
        "Aim defensive returns to the net or deep — avoid half-court lifts.",
        "Stay low and wide to cover more court when defending.",
    ],
}
_FALLBACK_GENERIC = [
    "Focus on getting to the shuttle early — preparation is half the battle.",
    "Keep a balanced ready position between shots to react faster.",
    "Watch the shuttle onto your racket and move your feet before your arm.",
]
_fallback_counters: dict[str, int] = {}

_BREAK_ENCOURAGEMENTS = [
    "Nice effort so far — use this break to reset and stay loose.",
    "Good work out there. Shake it off and get ready for the next rally.",
    "Take a breath, stay focused. You're building good habits.",
    "Solid session — keep that intensity when play picks back up.",
    "Stay light on your feet during the break, you're doing well.",
]
_break_enc_idx = 0

_SESSION_END_MESSAGES = [
    "Great session! Every rep on court makes you sharper — see you next time.",
    "That's a wrap. You put in the work today, and it shows.",
    "Session complete. Keep hitting the court and the improvements will stack up.",
    "Well played. Review your timeline and come back ready to level up.",
    "Good stuff today. Rest up and bring that energy back next session.",
]
_session_end_idx = 0


def _get_fallback_tip(stroke: str) -> str:
    tips = _FALLBACK_TIPS.get(stroke, _FALLBACK_GENERIC)
    idx = _fallback_counters.get(stroke, 0) % len(tips)
    _fallback_counters[stroke] = idx + 1
    return tips[idx]


def _result_signature(result: dict) -> str:
    m = result.get("metrics", {})
    return (
        f"{result.get('label', '?')}|"
        f"{m.get('technique', {}).get('label', '?')}|"
        f"{m.get('placement', {}).get('label', '?')}|"
        f"{m.get('quality', '?')}"
    )


async def _generate_commentary(result: dict) -> str:
    """Cached Gemini coaching tip with fallback to canned tips."""
    global _gemini_backoff_until

    sig = _result_signature(result)
    if sig in _commentary_cache:
        return _commentary_cache[sig]

    stroke = result.get("label", "Other")

    if not state.gemini_enabled or not state.gemini_client:
        return _get_fallback_tip(stroke)

    if time.monotonic() < _gemini_backoff_until:
        return _get_fallback_tip(stroke)

    try:
        m = result.get("metrics", {})
        prompt = (
            "You are a live badminton coach giving real-time courtside feedback. "
            "Give ONE sentence (15-20 words) — a specific, actionable technical tip only. "
            "No encouragement, no praise, no filler. Just the correction or suggestion. "
            "No lists, no markdown.\n\n"
            f"Stroke: {stroke}, "
            f"Technique: {m.get('technique', {}).get('label', '?')}, "
            f"Placement: {m.get('placement', {}).get('label', '?')}, "
            f"Position: {m.get('position', {}).get('label', '?')}, "
            f"Intent: {m.get('intent', {}).get('label', '?')}, "
            f"Quality: {m.get('quality', '?')}"
        )
        resp = await asyncio.to_thread(
            state.gemini_client.models.generate_content,
            model=state.gemini_model_name,
            contents=prompt,
        )

        text = ""
        if resp.candidates:
            candidate = resp.candidates[0]
            if candidate.content and candidate.content.parts:
                text = resp.text.strip().lstrip("*-•").strip()

        if not text:
            reason = "no_candidates"
            if resp.candidates:
                reason = getattr(resp.candidates[0], "finish_reason", "unknown")
            print(f"[live] gemini: empty response ({reason}) — using fallback")
            return _get_fallback_tip(stroke)

        if len(_commentary_cache) >= _COMMENTARY_CACHE_MAX:
            _commentary_cache.pop(next(iter(_commentary_cache)))
        _commentary_cache[sig] = text
        return text
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            _gemini_backoff_until = time.monotonic() + _GEMINI_BACKOFF_SECONDS
            print(f"[live] gemini: rate limited — backing off {_GEMINI_BACKOFF_SECONDS}s")
        elif "503" in msg or "UNAVAILABLE" in msg:
            _gemini_backoff_until = time.monotonic() + _GEMINI_BACKOFF_SECONDS
            print(f"[live] gemini: 503 unavailable — backing off {_GEMINI_BACKOFF_SECONDS}s")
        else:
            print(f"[live] gemini exception: {e}")
        return _get_fallback_tip(stroke)


@router.websocket("/sessions/{session_id}/ws")
async def live_session_ws(ws: WebSocket, session_id: str):
    """
    WebSocket live analysis — frame ingestion and inference are decoupled.
    """
    async with state._live_lock:
        if session_id not in state._live_sessions:
            await ws.close(code=4004, reason="Unknown session")
            return

    await ws.accept()
    print(f"[live] session {session_id} — WebSocket accepted, waiting for frames")
    await ws.send_json({"event": "status", "message": "Warming up — collecting initial frames..."})

    ring: list[np.ndarray] = []
    ring_lock = asyncio.Lock()
    bg_tasks: set[asyncio.Task] = set()
    last_commentary_sig: str | None = None
    closed = asyncio.Event()
    frame_count = 0

    async def _send_commentary(result: dict):
        try:
            text = await _generate_commentary(result)
            if text:
                print(f"[live] gemini: {text[:80]}")
                await ws.send_json({"event": "commentary", "text": text})
            else:
                print("[live] gemini: (empty response)")
        except Exception as e:
            print(f"[live] commentary error: {e}")

    async def _inference_loop():
        nonlocal last_commentary_sig
        first_sent = False
        was_break = False
        cycle = 0
        pose_passed = False
        while not closed.is_set():
            await asyncio.sleep(INFERENCE_INTERVAL_S)
            if closed.is_set():
                break
            try:
                async with ring_lock:
                    n = len(ring)
                    if n < RING_SIZE:
                        continue
                    snapshot = list(ring[-RING_SIZE:])

                if not first_sent:
                    print(f"[live] first inference — ring has {n} frames")
                    await ws.send_json({"event": "status", "message": "Analyzing..."})
                    first_sent = True

                run_pose = (cycle == 0) or (cycle % POSE_CHECK_INTERVAL == 0) or (was_break and not pose_passed)
                if run_pose:
                    step = max(1, len(snapshot) // POSE_SAMPLE_COUNT)
                    pose_sample = snapshot[::step][:POSE_SAMPLE_COUNT]
                    pose_conf = await asyncio.to_thread(_run_pose_check, pose_sample)
                    pose_passed = pose_conf >= POSE_CONF_MIN
                    if not pose_passed:
                        if not was_break:
                            print(f"[live] not badminton — pose gate failed (pose_conf={pose_conf:.3f})")
                            await ws.send_json({
                                "event": "break",
                                "reason": "no_badminton",
                                "message": "No badminton activity detected — point your camera at the court.",
                            })
                            was_break = True
                        cycle += 1
                        continue

                result = await asyncio.to_thread(_run_window_inference, snapshot)

                if result["badminton_prob"] < BADMINTON_PROB_MIN:
                    if not was_break:
                        global _break_enc_idx
                        print(f"[live] break detected (badminton_prob={result['badminton_prob']:.2f})")
                        await ws.send_json({
                            "event": "break",
                            "reason": "game_break",
                            "message": "Break in play detected — waiting for action to resume.",
                        })
                        enc = _BREAK_ENCOURAGEMENTS[_break_enc_idx % len(_BREAK_ENCOURAGEMENTS)]
                        _break_enc_idx += 1
                        await ws.send_json({"event": "commentary", "text": enc})
                        was_break = True
                    cycle += 1
                    continue

                was_break = False
                await ws.send_json({"event": "analysis", **result})

                sig = _result_signature(result)
                if sig != last_commentary_sig:
                    last_commentary_sig = sig
                    task = asyncio.create_task(_send_commentary(result))
                    bg_tasks.add(task)
                    task.add_done_callback(bg_tasks.discard)
            except Exception as e:
                print(f"[live] inference error: {e}")
            cycle += 1

    inference_task = asyncio.create_task(_inference_loop())

    try:
        while True:
            raw = await ws.receive_bytes()
            bgr = _decode_frame(raw)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb, (224, 224))
            async with ring_lock:
                ring.append(rgb_resized)
                frame_count += 1
                if len(ring) > RING_SIZE * 2:
                    ring[:] = ring[-RING_SIZE:]
            if frame_count in (1, RING_SIZE, RING_SIZE * 2):
                print(f"[live] frames received: {frame_count}")
    except WebSocketDisconnect:
        print(f"[live] client disconnected ({frame_count} frames received)")
    except Exception as e:
        print(f"[live] receive error: {e}")
        try:
            await ws.send_json({"event": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        closed.set()
        inference_task.cancel()
        for t in bg_tasks:
            t.cancel()
        if frame_count > RING_SIZE:
            try:
                global _session_end_idx
                msg = _SESSION_END_MESSAGES[_session_end_idx % len(_SESSION_END_MESSAGES)]
                _session_end_idx += 1
                await ws.send_json({"event": "commentary", "text": msg})
            except Exception:
                pass
        print(f"[live] session {session_id} cleaned up ({frame_count} total frames)")
        async with state._live_lock:
            if session_id in state._live_sessions:
                del state._live_sessions[session_id]
                state._live_semaphore.release()
