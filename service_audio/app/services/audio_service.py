import uuid
import subprocess
import shutil
from pathlib import Path
from fastapi import HTTPException

# Anchor data directory to the project root (absolute path).
# This file lives at:  <project_root>/service_audio/app/services/audio_service.py
# So going up 4 levels gives the project root.
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent.parent   # service_audio/app/services → project root
AUDIO_DIR = PROJECT_ROOT / "data" / "extracted_audio"


def extract_audio(video_path: str) -> str:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    audio_file = f"{uuid.uuid4()}.wav"
    audio_path = AUDIO_DIR / audio_file

    if shutil.which("ffmpeg") is None:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg is not installed or not in the system PATH."
        )

    command = [
        "ffmpeg",
        "-i", str(Path(video_path).resolve()),   # always pass absolute path to ffmpeg
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path)
    ]

    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"FFmpeg failed: {result.stderr.decode(errors='replace')}"
        )

    return str(audio_path)
