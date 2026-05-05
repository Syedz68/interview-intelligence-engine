"""
service_audio  –  Audio processing microservice
================================================
Runs on port 8001.
Handles: audio extraction, speech-to-text (Whisper),
         speaker diarization (pyannote), and speech analysis.

Environment requirements
------------------------
Uses .venv_audio  (numpy<2, required by pyannote/numba/torch 2.2.x).

Start
-----
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError
import logging
import tempfile
import os
from pathlib import Path

from app.services.audio_service import extract_audio
from app.services.stt_service import transcribe_audio
from app.services.diarization_service import diarize_audio
from app.services.speech_analysis_service import analyze_speech

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Interview Intelligence – Audio Service",
    version="1.0.0",
    description="Handles audio extraction, STT, diarization, and speech quality analysis.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"service": "audio", "status": "ok", "port": 8001}


@app.post("/extract-audio")
async def api_extract_audio(video_path: str = Form(...)):
    """Extract WAV audio from a local video file path."""
    try:
        audio_path = extract_audio(video_path)
        return {"audio_path": audio_path}
    except Exception as exc:
        logger.exception("Audio extraction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/transcribe")
async def api_transcribe(audio_path: str = Form(...)):
    """Run Whisper STT on a local audio file path."""
    try:
        result = transcribe_audio(audio_path)
        return result
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/diarize")
async def api_diarize(
    audio_path: str = Form(...),
    whisper_segments_json: str = Form(...),
    num_speakers: int | None = Form(None),
    max_speakers: int = Form(5),
):
    """
    Run speaker diarization on a local audio file.
    whisper_segments_json: JSON-encoded list of Whisper segments.
    """
    import json
    try:
        segments = json.loads(whisper_segments_json)
        result = diarize_audio(
            audio_path=audio_path,
            whisper_segments=segments,
            num_speakers=num_speakers,
            max_speakers=max_speakers,
        )
        return result
    except Exception as exc:
        logger.exception("Diarization failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze-speech")
async def api_analyze_speech(
    transcript: str = Form(...),
    segments_json: str = Form(...),
):
    """Compute speech quality metrics from a transcript + Whisper segments."""
    import json
    try:
        segments = json.loads(segments_json)
        result = analyze_speech(transcript=transcript, segments=segments)
        return result
    except Exception as exc:
        logger.exception("Speech analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/full-audio-pipeline")
async def full_audio_pipeline(
    video_path: str = Form(...),
    num_speakers: int | None = Form(None),
    skip_diarization: bool = Form(False),
    max_speakers: int = Form(5),
):
    """
    Convenience endpoint: runs all audio stages in sequence.
    Returns: audio_path, stt_result, diarization_result, speech_metrics.
    """
    import json

    try:
        audio_path = extract_audio(video_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio extraction failed: {exc}")

    try:
        stt_result = transcribe_audio(audio_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    if skip_diarization:
        from app.services.diarization_service import _fallback_result
        diarization_result = _fallback_result(stt_result["segments"])
    else:
        try:
            diarization_result = diarize_audio(
                audio_path=audio_path,
                whisper_segments=stt_result["segments"],
                num_speakers=num_speakers,
                max_speakers=max_speakers,
            )
        except Exception as exc:
            logger.warning("Diarization failed, using fallback: %s", exc)
            from app.services.diarization_service import _fallback_result
            diarization_result = _fallback_result(stt_result["segments"])

    candidate_transcript = (
        diarization_result.get("candidate_transcript") or stt_result["text"]
    )
    candidate_segments = (
        diarization_result.get("candidate_segments") or stt_result["segments"]
    )

    speech_metrics = analyze_speech(
        transcript=candidate_transcript,
        segments=candidate_segments,
    )

    return {
        "audio_path": audio_path,
        "stt": stt_result,
        "diarization": diarization_result,
        "speech_metrics": speech_metrics,
        "candidate_transcript": candidate_transcript,
        "candidate_segments": candidate_segments,
    }
