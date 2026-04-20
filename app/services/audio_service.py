import uuid
import subprocess
from pathlib import Path

AUDIO_DIR = Path("data/extracted_audio")


def extract_audio(video_path: str) -> str:

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    audio_file = f"{uuid.uuid4()}.wav"
    audio_path = AUDIO_DIR / audio_file

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",       # sample rate (good for Whisper)
        "-ac", "1",           # mono
        str(audio_path)
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return str(audio_path)