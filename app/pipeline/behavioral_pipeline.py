"""
behavioral_pipeline.py
----------------------
Orchestrates the full behavioral analysis pipeline.

Flow
----
1. Extract frames (frame_extraction_service)
2. Detect face landmarks (face_landmark_service)
3. Gaze + head-pose analysis (gaze_service)
4. Motion analysis (motion_service)
5. Emotion detection (emotion_service)
6. Speech transcription (stt_service via existing audio pipeline)
7. Speech quality analysis (speech_analysis_service)
8. Aggregate → scored output (behavioral_aggregator)

The pipeline is intentionally synchronous so it can be run inside a
FastAPI BackgroundTask or via Celery without any async complexity.
For production, wrap run_pipeline in a Celery task or FastAPI
background task and return a job_id immediately to the caller.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def run_pipeline(
    video_path: str,
    audio_path: str,
    *,
    fps: int = 2,
    skip_emotion: bool = False,
) -> dict[str, Any]:
    """
    Run the complete behavioral analysis pipeline.

    Parameters
    ----------
    video_path : str
        Local path to the input video file.
    audio_path : str
        Local path to the extracted audio WAV file.
    fps : int
        Frame extraction rate (1-5 recommended).
    skip_emotion : bool
        Set True to skip DeepFace (useful in CI / lightweight deployments).

    Returns
    -------
    Full scored result dict compatible with BehavioralAnalysisResult schema.
    """
    timings: dict[str, float] = {}

    # ── 1. Frame extraction ───────────────────────────────────────────────────
    logger.info("Pipeline: extracting frames at %d fps", fps)
    t0 = time.perf_counter()
    from app.services.frame_extraction_service import extract_frames
    frames = extract_frames(video_path, fps=fps)
    timings["frame_extraction"] = round(time.perf_counter() - t0, 2)
    logger.info("Pipeline: extracted %d frames", len(frames))

    # ── 2. Face landmark detection ────────────────────────────────────────────
    logger.info("Pipeline: detecting face landmarks")
    t0 = time.perf_counter()
    from app.services.face_landmark_service import extract_face_landmarks
    processed_frames = extract_face_landmarks(frames)
    timings["landmark_detection"] = round(time.perf_counter() - t0, 2)

    # Attach frame_path back into processed_frames for gaze/pose loading
    for pf, f in zip(processed_frames, frames):
        pf.setdefault("frame_path", f["frame_path"])

    # ── 3. Gaze + head-pose analysis ──────────────────────────────────────────
    logger.info("Pipeline: analysing gaze & head pose")
    t0 = time.perf_counter()
    from app.services.gaze_service import analyze_gaze
    gaze_metrics = analyze_gaze(processed_frames)
    timings["gaze_analysis"] = round(time.perf_counter() - t0, 2)

    # ── 4. Motion analysis ────────────────────────────────────────────────────
    logger.info("Pipeline: analysing motion patterns")
    t0 = time.perf_counter()
    from app.services.motion_service import analyze_motion
    motion_metrics = analyze_motion(processed_frames)
    timings["motion_analysis"] = round(time.perf_counter() - t0, 2)

    # ── 5. Emotion detection ──────────────────────────────────────────────────
    emotion_metrics: dict[str, Any] = {}
    if not skip_emotion:
        logger.info("Pipeline: running emotion detection (DeepFace)")
        t0 = time.perf_counter()
        try:
            from app.services.emotion_service import analyze_emotions
            emotion_metrics = analyze_emotions(frames)
        except Exception as exc:
            logger.warning("Pipeline: emotion detection failed – %s", exc)
            emotion_metrics = {}
        timings["emotion_detection"] = round(time.perf_counter() - t0, 2)
    else:
        logger.info("Pipeline: emotion detection skipped")

    # ── 6. Speech transcription (already done upstream; we just re-use) ───────
    logger.info("Pipeline: transcribing audio")
    t0 = time.perf_counter()
    from app.services.stt_service import transcribe_audio
    stt_result = transcribe_audio(audio_path)
    timings["transcription"] = round(time.perf_counter() - t0, 2)

    # ── 7. Speech quality analysis ────────────────────────────────────────────
    logger.info("Pipeline: analysing speech quality")
    t0 = time.perf_counter()
    from app.services.speech_analysis_service import analyze_speech
    speech_metrics = analyze_speech(
        transcript=stt_result["text"],
        segments=stt_result["segments"],
    )
    timings["speech_analysis"] = round(time.perf_counter() - t0, 2)

    # ── 8. Aggregate → final scores ───────────────────────────────────────────
    logger.info("Pipeline: aggregating scores")
    from app.services.behavioral_aggregator import aggregate_behavioral_scores
    scored = aggregate_behavioral_scores(
        gaze_metrics=gaze_metrics,
        motion_metrics=motion_metrics,
        emotion_metrics=emotion_metrics,
        speech_metrics=speech_metrics,
    )

    # ── Compose full result ───────────────────────────────────────────────────
    # sanitize() converts any numpy.float32/float64/int* that leaked from
    # cv2 / MediaPipe / DeepFace into plain Python types so Pydantic v2 can
    # serialize the response without PydanticSerializationError.
    from app.core.numpy_encoder import sanitize

    return sanitize({
        **scored,
        "transcript": stt_result["text"],
        "segments": stt_result["segments"],
        "raw_metrics": {
            "gaze": gaze_metrics,
            "motion": motion_metrics,
            "emotion": emotion_metrics,
            "speech": speech_metrics,
        },
        "timings": timings,
    })