from fastapi import APIRouter, UploadFile, File, Form
from app.services.video_service import download_video_from_url, save_uploaded_video
from app.scehmas.request import VideoUploadRequest

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

async def process_video(
    video_url: str = Form(None),
    file: UploadFile = File(None)
):

    if not video_url and not file:
        return {"error": "Provide either video_url or file"}

    if video_url:
        video_path = download_video_from_url(video_url)
    else:
        video_path = save_uploaded_video(file)


    return {
        "video_path": video_path,
        "audio_path": None
    }