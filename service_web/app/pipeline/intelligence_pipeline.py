"""
intelligence_pipeline.py
-------------------------
Full Interview Intelligence Pipeline
Orchestrates all stages across both microservices.

Stage 1  → Video Ingestion
Stage 2  → Audio Extraction          ← service_audio (HTTP)
Stage 3  → Frame Extraction
Stage 4  → Face Landmark Detection
Stage 5  → Gaze + Head-Pose Analysis
Stage 6  → Motion Analysis
Stage 7  → Emotion Detection (DeepFace)
Stage 8  → Speech Transcription      ← service_audio (HTTP)
Stage 9  → Speaker Diarization       ← service_audio (HTTP)
Stage 10 → Speech Quality Analysis   ← service_audio (HTTP)
Stage 11 → English Fluency Analysis (Ollama AI)
Stage 12 → Answer Relevance Analysis (Ollama AI)
Stage 13 → Behavioral Aggregation
Stage 14 → Behavioral Report Generation (Ollama AI)

Audio stages (8-10) are delegated to service_audio on port 8001.
All vision and AI stages run inside service_web.
"""

from __future__ import annotations
import json
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

AUDIO_SERVICE_URL = os.getenv("AUDIO_SERVICE_URL", "http://localhost:8001")


def _audio_post(endpoint: str, data: dict) -> dict:
    url = f"{AUDIO_SERVICE_URL}{endpoint}"
    try:
        resp = requests.post(url, data=data, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach service_audio at {AUDIO_SERVICE_URL}. "
            "Make sure it is running (port 8001)."
        ) from exc


def run_full_pipeline(
    video_path: str,
    audio_path: str,
    *,
    fps: int = 2,
    skip_emotion: bool = False,
    skip_diarization: bool = False,
    job_role: str | None = None,
    candidate_name: str | None = None,
    num_speakers: int | None = None,
    max_speakers: int = 5,
) -> dict[str, Any]:
    timings: dict[str, float] = {}

    # Stage 3: Frame extraction
    logger.info("[Pipeline] Stage 3 – Extracting frames at %d fps", fps)
    t0 = time.perf_counter()
    from app.services.frame_extraction_service import extract_frames
    frames = extract_frames(video_path, fps=fps)
    timings["frame_extraction"] = round(time.perf_counter() - t0, 2)
    logger.info("[Pipeline] Extracted %d frames", len(frames))

    # Stage 4: Face landmarks
    logger.info("[Pipeline] Stage 4 – Detecting face landmarks")
    t0 = time.perf_counter()
    from app.services.face_landmark_service import extract_face_landmarks
    processed_frames = extract_face_landmarks(frames)
    for pf, f in zip(processed_frames, frames):
        pf.setdefault("frame_path", f["frame_path"])
    timings["landmark_detection"] = round(time.perf_counter() - t0, 2)

    # Stage 5: Gaze + head pose
    logger.info("[Pipeline] Stage 5 – Gaze & head-pose analysis")
    t0 = time.perf_counter()
    from app.services.gaze_service import analyze_gaze
    gaze_metrics = analyze_gaze(processed_frames)
    timings["gaze_analysis"] = round(time.perf_counter() - t0, 2)

    # Stage 6: Motion analysis
    logger.info("[Pipeline] Stage 6 – Motion analysis")
    t0 = time.perf_counter()
    from app.services.motion_service import analyze_motion
    motion_metrics = analyze_motion(processed_frames)
    timings["motion_analysis"] = round(time.perf_counter() - t0, 2)

    # Stage 7: Emotion detection
    emotion_metrics: dict[str, Any] = {}
    if not skip_emotion:
        logger.info("[Pipeline] Stage 7 – Emotion detection (DeepFace)")
        t0 = time.perf_counter()
        try:
            from app.services.emotion_service import analyze_emotions
            emotion_metrics = analyze_emotions(frames)
        except Exception as exc:
            logger.warning("[Pipeline] Emotion detection failed: %s", exc)
        timings["emotion_detection"] = round(time.perf_counter() - t0, 2)
    else:
        logger.info("[Pipeline] Stage 7 – Emotion detection skipped")

    # Stage 8: Speech transcription (service_audio)
    logger.info("[Pipeline] Stage 8 – Whisper transcription (service_audio)")
    t0 = time.perf_counter()
    stt_result = _audio_post("/transcribe", {"audio_path": audio_path})
    timings["transcription"] = round(time.perf_counter() - t0, 2)
    logger.info("[Pipeline] Transcribed %d words", len(stt_result["text"].split()))

    # Stage 9: Speaker diarization (service_audio)
    diarization_result: dict[str, Any] = {}
    if not skip_diarization:
        logger.info("[Pipeline] Stage 9 – Speaker diarization (service_audio)")
        t0 = time.perf_counter()
        payload = {
            "audio_path": audio_path,
            "whisper_segments_json": json.dumps(stt_result["segments"]),
            "max_speakers": max_speakers,
        }
        if num_speakers is not None:
            payload["num_speakers"] = num_speakers
        diarization_result = _audio_post("/diarize", payload)
        timings["diarization"] = round(time.perf_counter() - t0, 2)
    else:
        full_text = stt_result["text"]
        segs = stt_result["segments"]
        diarization_result = {
            "turns": [{"speaker": "SPEAKER_00", "role": "candidate",
                       "start": segs[0]["start"] if segs else 0.0,
                       "end": segs[-1]["end"] if segs else 0.0,
                       "text": full_text, "word_count": len(full_text.split())}] if segs else [],
            "speaker_map": {"SPEAKER_00": "candidate"},
            "role_confidence": "N/A – diarization skipped",
            "candidate_transcript": full_text,
            "interviewer_transcript": "",
            "candidate_segments": segs,
            "interviewer_segments": [],
            "diarization_available": False,
        }
        logger.info("[Pipeline] Stage 9 – Diarization skipped")

    candidate_transcript = diarization_result.get("candidate_transcript") or stt_result["text"]
    candidate_segments = diarization_result.get("candidate_segments") or stt_result["segments"]

    # Stage 10: Speech quality analysis (service_audio)
    logger.info("[Pipeline] Stage 10 – Speech quality analysis (service_audio)")
    t0 = time.perf_counter()
    speech_metrics = _audio_post("/analyze-speech", {
        "transcript": candidate_transcript,
        "segments_json": json.dumps(candidate_segments),
    })
    timings["speech_analysis"] = round(time.perf_counter() - t0, 2)

    # Stage 11: English fluency (Ollama)
    logger.info("[Pipeline] Stage 11 – English fluency analysis (AI)")
    t0 = time.perf_counter()
    from app.services.english_fluency_service import analyze_english_fluency
    fluency_report = analyze_english_fluency(
        candidate_transcript=candidate_transcript,
        speech_metrics=speech_metrics,
    )
    timings["fluency_analysis"] = round(time.perf_counter() - t0, 2)

    # Stage 12: Answer relevance (Ollama)
    logger.info("[Pipeline] Stage 12 – Answer relevance analysis (AI)")
    t0 = time.perf_counter()
    from app.services.answer_relevance_service import analyze_answer_relevance
    qa_report = analyze_answer_relevance(
        diarization_result=diarization_result,
        job_role=job_role,
    )
    timings["qa_analysis"] = round(time.perf_counter() - t0, 2)

    # Stage 13: Behavioral aggregation
    logger.info("[Pipeline] Stage 13 – Behavioral score aggregation")
    from app.services.behavioral_aggregator import aggregate_behavioral_scores
    behavioral_scores = aggregate_behavioral_scores(
        gaze_metrics=gaze_metrics,
        motion_metrics=motion_metrics,
        emotion_metrics=emotion_metrics,
        speech_metrics=speech_metrics,
    )

    # Stage 14: Behavioral report (Ollama)
    logger.info("[Pipeline] Stage 14 – Behavioral report generation (AI)")
    t0 = time.perf_counter()
    from app.services.behavioral_report_service import generate_behavioral_report
    behavioral_report = generate_behavioral_report(
        behavioral_scores=behavioral_scores,
        speech_metrics=speech_metrics,
        fluency_report=fluency_report,
        qa_report=qa_report,
        job_role=job_role,
        candidate_name=candidate_name,
    )
    timings["behavioral_report"] = round(time.perf_counter() - t0, 2)

    from app.core.numpy_encoder import sanitize
    return sanitize({
        **behavioral_scores,
        "transcript": stt_result["text"],
        "candidate_transcript": candidate_transcript,
        "segments": stt_result["segments"],
        "candidate_segments": candidate_segments,
        "diarization": {
            "available": diarization_result.get("diarization_available", False),
            "speaker_map": diarization_result.get("speaker_map", {}),
            "role_confidence": diarization_result.get("role_confidence", "N/A"),
            "turns": diarization_result.get("turns", []),
        },
        "english_fluency": fluency_report,
        "answer_relevance": qa_report,
        "behavioral_report": behavioral_report,
        "raw_metrics": {
            "gaze": gaze_metrics,
            "motion": motion_metrics,
            "emotion": emotion_metrics,
            "speech": speech_metrics,
        },
        "timings": timings,
    })
