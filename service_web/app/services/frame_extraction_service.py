import cv2
import os
import uuid
from pathlib import Path

# Anchor to project root (absolute path).
# This file lives at: <project_root>/service_web/app/services/frame_extraction_service.py
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent.parent   # → project root
FRAME_DIR = PROJECT_ROOT / "data" / "frames"


def extract_frames(video_path: str, fps: int = 2):
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve to absolute path so OpenCV can always find the file
    video_path = str(Path(video_path).resolve())

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))

    frames = []
    frame_count = 0

    video_id = str(uuid.uuid4())
    video_frame_dir = FRAME_DIR / video_id
    video_frame_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps

            frame_filename = f"frame_{frame_count}.jpg"
            frame_path = video_frame_dir / frame_filename

            cv2.imwrite(str(frame_path), frame)

            frames.append({
                "frame_path": str(frame_path),
                "timestamp": round(timestamp, 2)
            })

        frame_count += 1

    cap.release()

    return frames
