"""
intelligence_pipeline.py
-------------------------
Full Interview Intelligence Pipeline — v3

Orchestrates all stages:

Stage 1 — Video Ingestion
Stage 2 — Audio Extraction
Stage 3 — Frame Extraction
Stage 4 — Face Landmark Detection
Stage 5 — Gaze + Head-Pose Analysis
Stage 6 — Motion Analysis
Stage 7 — Emotion Detection (DeepFace)
Stage 8 — Speech Transcription (Whisper)
Stage 9 — Speaker Diarization (pyannote)        ← NEW
Stage 10 — Speech Quality Analysis
Stage 11 — English Fluency Analysis (Claude AI)  ← NEW
Stage 12 — Answer Relevance Analysis (Claude AI) ← NEW
Stage 13 — Behavioral Aggregation
Stage 14 — Behavioral Report Generation (Claude AI) ← NEW

The pipeline is intentionally synchronous. Wrap in Celery/BackgroundTask
for async use and return a job_id immediately to the caller.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


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
) -> dict[str, Any]:
    """
    Run the complete Interview Intelligence pipeline.

    Parameters
    ----------
    video_path        : local path to the input video
    audio_path        : local path to the extracted 16kHz mono WAV
    fps               : frame extraction rate (1-5 recommended)
    skip_emotion      : skip DeepFace emotion detection (faster)
    skip_diarization  : skip pyannote diarization (faster, less accurate Q&A split)
    job_role          : e.g. "Senior Backend Engineer" – improves AI assessments
    candidate_name    : for personalised report
    num_speakers      : exact speaker count hint for pyannote (None = auto)

    Returns
    -------
    Full scored + narrated intelligence result dict.
    """
    timings: dict[str, float] = {}

    # ── Stage 3: Frame extraction ──────────────────────────────────────────────
    logger.info("[Pipeline] Stage 3 – Extracting frames at %d fps", fps)
    t0 = time.perf_counter()
    from app.services.frame_extraction_service import extract_frames
    frames = extract_frames(video_path, fps=fps)
    timings["frame_extraction"] = round(time.perf_counter() - t0, 2)
    logger.info("[Pipeline] Extracted %d frames", len(frames))

    # ── Stage 4: Face landmarks ────────────────────────────────────────────────
    logger.info("[Pipeline] Stage 4 – Detecting face landmarks")
    t0 = time.perf_counter()
    from app.services.face_landmark_service import extract_face_landmarks
    processed_frames = extract_face_landmarks(frames)
    for pf, f in zip(processed_frames, frames):
        pf.setdefault("frame_path", f["frame_path"])
    timings["landmark_detection"] = round(time.perf_counter() - t0, 2)

    # ── Stage 5: Gaze + head pose ──────────────────────────────────────────────
    logger.info("[Pipeline] Stage 5 – Gaze & head-pose analysis")
    t0 = time.perf_counter()
    from app.services.gaze_service import analyze_gaze
    gaze_metrics = analyze_gaze(processed_frames)
    timings["gaze_analysis"] = round(time.perf_counter() - t0, 2)

    # ── Stage 6: Motion analysis ───────────────────────────────────────────────
    logger.info("[Pipeline] Stage 6 – Motion analysis")
    t0 = time.perf_counter()
    from app.services.motion_service import analyze_motion
    motion_metrics = analyze_motion(processed_frames)
    timings["motion_analysis"] = round(time.perf_counter() - t0, 2)

    # ── Stage 7: Emotion detection ─────────────────────────────────────────────
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

    # ── Stage 8: Speech transcription (Whisper) ────────────────────────────────
    logger.info("[Pipeline] Stage 8 – Whisper transcription")
    t0 = time.perf_counter()
    from app.services.stt_service import transcribe_audio
    stt_result = transcribe_audio(audio_path)
    timings["transcription"] = round(time.perf_counter() - t0, 2)
    logger.info("[Pipeline] Transcribed %d words", len(stt_result["text"].split()))

    # ── Stage 9: Speaker diarization (pyannote) ────────────────────────────────
    diarization_result: dict[str, Any] = {}
    if not skip_diarization:
        logger.info("[Pipeline] Stage 9 – Speaker diarization")
        t0 = time.perf_counter()
        from app.services.diarization_service import diarize_audio
        diarization_result = diarize_audio(
            audio_path=audio_path,
            whisper_segments=stt_result["segments"],
            num_speakers=num_speakers,
        )
        timings["diarization"] = round(time.perf_counter() - t0, 2)
        logger.info(
            "[Pipeline] Diarization: available=%s, turns=%d",
            diarization_result.get("diarization_available"),
            len(diarization_result.get("turns", [])),
        )
    else:
        # Build a fallback diarization result so downstream services still work
        from app.services.diarization_service import _fallback_result
        diarization_result = _fallback_result(stt_result["segments"])
        logger.info("[Pipeline] Stage 9 – Diarization skipped")

    # Resolve candidate transcript (diarized or full)
    candidate_transcript = (
        diarization_result.get("candidate_transcript") or stt_result["text"]
    )
    candidate_segments = (
        diarization_result.get("candidate_segments") or stt_result["segments"]
    )

    # ── Stage 10: Speech quality analysis ─────────────────────────────────────
    logger.info("[Pipeline] Stage 10 – Speech quality analysis")
    t0 = time.perf_counter()
    from app.services.speech_analysis_service import analyze_speech
    speech_metrics = analyze_speech(
        transcript=candidate_transcript,
        segments=candidate_segments,
    )
    timings["speech_analysis"] = round(time.perf_counter() - t0, 2)

    # ── Stage 11: English fluency analysis (Claude AI) ─────────────────────────
    logger.info("[Pipeline] Stage 11 – English fluency analysis (AI)")
    t0 = time.perf_counter()
    from app.services.english_fluency_service import analyze_english_fluency
    fluency_report = analyze_english_fluency(
        candidate_transcript=candidate_transcript,
        speech_metrics=speech_metrics,
    )
    timings["fluency_analysis"] = round(time.perf_counter() - t0, 2)
    logger.info(
        "[Pipeline] Fluency: band=%s, score=%s, ai=%s",
        fluency_report.get("overall_band"),
        fluency_report.get("overall_score"),
        fluency_report.get("ai_powered"),
    )

    # ── Stage 12: Answer relevance analysis (Claude AI) ────────────────────────
    logger.info("[Pipeline] Stage 12 – Answer relevance analysis (AI)")
    t0 = time.perf_counter()
    from app.services.answer_relevance_service import analyze_answer_relevance
    qa_report = analyze_answer_relevance(
        diarization_result=diarization_result,
        job_role=job_role,
    )
    timings["qa_analysis"] = round(time.perf_counter() - t0, 2)
    logger.info(
        "[Pipeline] Q&A: pairs=%d, overall_score=%s, ai=%s",
        len(qa_report.get("qa_pairs", [])),
        qa_report.get("aggregate", {}).get("overall_qa_score"),
        qa_report.get("ai_powered"),
    )

    # ── Stage 13: Behavioral score aggregation ─────────────────────────────────
    logger.info("[Pipeline] Stage 13 – Behavioral score aggregation")
    from app.services.behavioral_aggregator import aggregate_behavioral_scores
    behavioral_scores = aggregate_behavioral_scores(
        gaze_metrics=gaze_metrics,
        motion_metrics=motion_metrics,
        emotion_metrics=emotion_metrics,
        speech_metrics=speech_metrics,
    )

    # ── Stage 14: Behavioral report generation (Claude AI) ─────────────────────
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
    logger.info(
        "[Pipeline] Report: recommendation=%s, score=%s, ai=%s",
        behavioral_report.get("hiring_recommendation"),
        behavioral_report.get("overall_behavioral_score"),
        behavioral_report.get("ai_powered"),
    )

    # ── Sanitize + compose final output ───────────────────────────────────────
    from app.core.numpy_encoder import sanitize

    return sanitize({
        # ── Core behavioral scores (1-5) ──────────────────────────────────────
        **behavioral_scores,

        # ── Transcription ─────────────────────────────────────────────────────
        "transcript": stt_result["text"],
        "candidate_transcript": candidate_transcript,
        "segments": stt_result["segments"],
        "candidate_segments": candidate_segments,

        # ── Diarization ───────────────────────────────────────────────────────
        "diarization": {
            "available": diarization_result.get("diarization_available", False),
            "speaker_map": diarization_result.get("speaker_map", {}),
            "turns": diarization_result.get("turns", []),
        },

        # ── AI analysis results ───────────────────────────────────────────────
        "english_fluency": fluency_report,
        "answer_relevance": qa_report,
        "behavioral_report": behavioral_report,

        # ── Raw metrics (debug/audit) ──────────────────────────────────────────
        "raw_metrics": {
            "gaze": gaze_metrics,
            "motion": motion_metrics,
            "emotion": emotion_metrics,
            "speech": speech_metrics,
        },

        # ── Timings ───────────────────────────────────────────────────────────
        "timings": timings,
    })
