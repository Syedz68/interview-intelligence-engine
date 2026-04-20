from fastapi import APIRouter, UploadFile, File, Form
from app.services.video_service import download_video_from_url, save_uploaded_video
from app.services.audio_service import extract_audio
from app.services.stt_service import transcribe_audio
from app.scehmas.request import VideoUploadRequest

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

@router.post("/process")
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

    audio_path = extract_audio(video_path)
    stt_result = transcribe_audio(audio_path)

    return {
        "video_path": video_path,
        "audio_path": audio_path,
        "transcript": stt_result["text"],
        "segments": stt_result["segments"]
    }