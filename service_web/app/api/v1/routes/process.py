import os
import requests
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.services.video_service import download_video_from_url, save_uploaded_video
from app.services.frame_extraction_service import extract_frames
from app.services.face_landmark_service import extract_face_landmarks
from app.schemas.request import VideoUploadRequest

AUDIO_SERVICE_URL = os.getenv("AUDIO_SERVICE_URL", "http://localhost:8001")

def _audio_post(endpoint: str, data: dict) -> dict:
    url = f"{AUDIO_SERVICE_URL}{endpoint}"
    try:
        resp = requests.post(url, data=data, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError as exc:
        raise HTTPException(status_code=503, detail=f"service_audio unreachable at {AUDIO_SERVICE_URL}. Is it running on port 8001?") from exc

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

@router.post("/process-interview")
async def process_interview(
    video_url: str | None = Form(None),
    file: UploadFile = File(None)
):
    if not video_url and not file:
        return {"error": "Provide either video_url or file"}

    if video_url:
        video_path = download_video_from_url(video_url)
    else:
        video_path = save_uploaded_video(file)

    audio_result = _audio_post("/extract-audio", {"video_path": video_path})
    audio_path = audio_result["audio_path"]
    stt_result = _audio_post("/transcribe", {"audio_path": audio_path})

    return {
        "video_path": video_path,
        "audio_path": audio_path,
        "transcript": stt_result["text"],
        "segments": stt_result["segments"]
    }

@router.post("/process-video")
async def process_video(
    video_url: str | None = Form(None),
    file: UploadFile = File(None)
):
    if not video_url and not file:
        return {"error": "Provide either video_url or file"}

    if video_url:
        video_path = download_video_from_url(video_url)
    else:
        video_path = save_uploaded_video(file)

    frames = extract_frames(video_path)
    processed_frames = extract_face_landmarks(frames)
    return processed_frames