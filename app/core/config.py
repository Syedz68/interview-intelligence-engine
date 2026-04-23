from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = Field(..., env="PROJECT_NAME")
    ENV: str = Field(..., env="ENV")
    VERSION: str = Field(..., env="VERSION")
    DEBUG: bool = Field(..., env="DEBUG")
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALLOWED_ORIGINS: list[str] = Field(..., env="ALLOWED_ORIGINS")

    # Primary Gemini key (required)
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    # Additional Gemini keys for rate-limit rotation (optional)
    GEMINI_API_KEY_2: Optional[str] = Field(None, env="GEMINI_API_KEY_2")
    GEMINI_API_KEY_3: Optional[str] = Field(None, env="GEMINI_API_KEY_3")

    HF_TOKEN: str = Field(..., env="HF_TOKEN")
    W_COMMUNICATION: float = Field(..., env="W_COMMUNICATION")
    W_ENGAGEMENT: float = Field(..., env="W_ENGAGEMENT")
    W_CONFIDENCE: float = Field(..., env="W_CONFIDENCE")
    W_PROFESSIONALISM: float = Field(..., env="W_PROFESSIONALISM")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
