import uuid
import requests
from pathlib import Path

DATA_DIR = Path("data/input_video")


def download_video_from_url(video_url: str) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_name = f"{uuid.uuid4()}.mp4"
    file_path = DATA_DIR / file_name

    response = requests.get(video_url, stream=True)

    if response.status_code != 200:
        raise Exception("Failed to download video")

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

    return str(file_path)


def save_uploaded_video(file) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = DATA_DIR / file_name

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return str(file_path)