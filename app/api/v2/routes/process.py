"""
process.py  (updated)
---------------------
API routes for the Interview Intelligence Engine.

New endpoint:  POST /intelligence/analyze-interview
    Runs the full behavioral analysis pipeline and returns a scored result.

Existing endpoints are preserved unchanged.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.schemas.response import BehavioralAnalysisResult
from app.services.audio_service import extract_audio
from app.services.face_landmark_service import extract_face_landmarks
from app.services.frame_extraction_service import extract_frames
from app.services.stt_service import transcribe_audio
from app.services.video_service import download_video_from_url, save_uploaded_video

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/new-intelligence", tags=["New Intelligence"])


# ── Existing endpoints (unchanged) ───────────────────────────────────────────

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

    audio_path = extract_audio(video_path)
    stt_result = transcribe_audio(audio_path)

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


# ── New: full behavioral analysis ────────────────────────────────────────────

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
    """
    Full pipeline:
    1. Ingest video (URL or upload)
    2. Extract audio → Whisper STT
    3. Extract frames → MediaPipe landmarks
    4. Gaze / head-pose analysis
    5. Motion analysis
    6. Emotion detection (DeepFace)
    7. Speech quality analysis
    8. Aggregate → scored output
    """
    if not video_url and not file:
        raise HTTPException(status_code=400, detail="Provide either video_url or file.")

    # ── Ingest ────────────────────────────────────────────────────────────────
    try:
        video_path = (
            download_video_from_url(video_url) if video_url else save_uploaded_video(file)
        )
    except Exception as exc:
        logger.exception("Video ingest failed")
        raise HTTPException(status_code=422, detail=f"Video ingest failed: {exc}") from exc

    # ── Audio extraction ──────────────────────────────────────────────────────
    try:
        audio_path = extract_audio(video_path)
    except Exception as exc:
        logger.exception("Audio extraction failed")
        raise HTTPException(status_code=500, detail=f"Audio extraction failed: {exc}") from exc

    # ── Pipeline ──────────────────────────────────────────────────────────────
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