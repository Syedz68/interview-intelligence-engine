"""
process.py  (updated)
---------------------
API routes for the Interview Intelligence Engine.
Audio work is delegated to service_audio (port 8001) via HTTP.
"""

from __future__ import annotations

import json
import logging
import os

import requests
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.response import BehavioralAnalysisResult, FullIntelligenceResult
from app.services.face_landmark_service import extract_face_landmarks
from app.services.frame_extraction_service import extract_frames
from app.services.video_service import download_video_from_url, save_uploaded_video

logger = logging.getLogger(__name__)

AUDIO_SERVICE_URL = os.getenv("AUDIO_SERVICE_URL", "http://localhost:8001")


def _audio_post(endpoint: str, data: dict) -> dict:
    """POST to service_audio and return JSON. Raises 503 if unreachable."""
    url = f"{AUDIO_SERVICE_URL}{endpoint}"
    try:
        resp = requests.post(url, data=data, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"service_audio unreachable at {AUDIO_SERVICE_URL}. Is it running on port 8001?"
        ) from exc


router = APIRouter(prefix="/new-intelligence", tags=["New Intelligence"])


# ── Existing endpoints ────────────────────────────────────────────────────────

@router.post("/process-interview")
async def process_interview(
    video_url: str | None = Form(None),
    file: UploadFile = File(None),
):
    if not video_url and not file:
        return {"error": "Provide either video_url or file"}

    video_path = (
        download_video_from_url(video_url) if video_url else save_uploaded_video(file)
    )

    audio_result = _audio_post("/extract-audio", {"video_path": video_path})
    audio_path = audio_result["audio_path"]
    stt_result = _audio_post("/transcribe", {"audio_path": audio_path})

    return {
        "video_path": video_path,
        "audio_path": audio_path,
        "transcript": stt_result["text"],
        "segments": stt_result["segments"],
    }


@router.post("/process-video")
async def process_video(
    video_url: str | None = Form(None),
    file: UploadFile = File(None),
):
    if not video_url and not file:
        return {"error": "Provide either video_url or file"}

    video_path = (
        download_video_from_url(video_url) if video_url else save_uploaded_video(file)
    )

    frames = extract_frames(video_path)
    processed_frames = extract_face_landmarks(frames)
    return processed_frames


# ── Full behavioral analysis (v2) ─────────────────────────────────────────────

@router.post(
    "/analyze-interview",
    response_model=BehavioralAnalysisResult,
    summary="Run full behavioral analysis pipeline",
    description=(
        "Accepts a video file or URL, runs the complete CV + NLP pipeline, "
        "and returns a scored behavioral assessment."
    ),
)
async def analyze_interview(
    video_url: str | None = Form(None),
    file: UploadFile = File(None),
    fps: int = Form(2, ge=1, le=5, description="Frame extraction rate"),
    skip_emotion: bool = Form(False, description="Skip DeepFace (faster, less accurate)"),
):
    if not video_url and not file:
        raise HTTPException(status_code=400, detail="Provide either video_url or file.")

    try:
        video_path = (
            download_video_from_url(video_url) if video_url else save_uploaded_video(file)
        )
    except Exception as exc:
        logger.exception("Video ingest failed")
        raise HTTPException(status_code=422, detail=f"Video ingest failed: {exc}") from exc

    # Delegate audio extraction to service_audio
    audio_result = _audio_post("/extract-audio", {"video_path": video_path})
    audio_path = audio_result["audio_path"]

    try:
        from app.pipeline.behavioral_pipeline import run_pipeline
        result = run_pipeline(
            video_path=video_path,
            audio_path=audio_path,
            fps=fps,
            skip_emotion=skip_emotion,
        )
    except Exception as exc:
        logger.exception("Pipeline execution failed")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    return result


# ── Full AI-powered intelligence pipeline (v3) ────────────────────────────────

@router.post(
    "/analyze-interview-v3",
    response_model=FullIntelligenceResult,
    summary="Full Interview Intelligence: diarization + fluency + Q&A + behavioral report",
    description=(
        "Runs the complete v3 pipeline:\n"
        "1. Video/audio ingest\n"
        "2. Frame analysis (gaze, motion, emotion)\n"
        "3. Whisper STT (via service_audio)\n"
        "4. pyannote speaker diarization (via service_audio)\n"
        "5. Ollama AI: English fluency & accuracy report\n"
        "6. Ollama AI: Answer relevance & correctness scoring\n"
        "7. Behavioral score aggregation\n"
        "8. Ollama AI: Narrative behavioral report with hiring recommendation\n\n"
        "Requires HF_TOKEN in .env for pyannote diarization.\n"
        "Requires Ollama running locally for AI stages."
    ),
)
async def analyze_interview_v3(
    video_url: str | None = Form(None),
    file: UploadFile = File(None),
    fps: int = Form(2, ge=1, le=5, description="Frame extraction rate"),
    skip_emotion: bool = Form(False, description="Skip DeepFace emotion detection"),
    skip_diarization: bool = Form(False, description="Skip pyannote diarization"),
    job_role: str | None = Form(None, description="Target job role for context-aware AI scoring"),
    candidate_name: str | None = Form(None, description="Candidate name for personalised report"),
    num_speakers: int | None = Form(None, description="Exact speaker count hint for pyannote"),
    max_speakers: int = Form(5, ge=2, le=5, description="Max speakers for auto-detection (2-5)"),
):
    if not video_url and not file:
        raise HTTPException(status_code=400, detail="Provide either video_url or file.")

    try:
        video_path = (
            download_video_from_url(video_url) if video_url else save_uploaded_video(file)
        )
    except Exception as exc:
        logger.exception("Video ingest failed")
        raise HTTPException(status_code=422, detail=f"Video ingest failed: {exc}") from exc

    # Delegate audio extraction to service_audio
    audio_result = _audio_post("/extract-audio", {"video_path": video_path})
    audio_path = audio_result["audio_path"]

    try:
        from app.pipeline.intelligence_pipeline import run_full_pipeline
        result = run_full_pipeline(
            video_path=video_path,
            audio_path=audio_path,
            fps=fps,
            skip_emotion=skip_emotion,
            skip_diarization=skip_diarization,
            job_role=job_role,
            candidate_name=candidate_name,
            num_speakers=num_speakers,
            max_speakers=max_speakers,
        )
    except Exception as exc:
        logger.exception("v3 pipeline execution failed")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    return result
