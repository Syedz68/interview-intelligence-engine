from typing import Optional, Dict
from pydantic import BaseModel

class VideoUploadRequest(BaseModel):
    video_url: str