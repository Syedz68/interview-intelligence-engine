from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str  = Field(..., env="PROJECT_NAME")
    ENV:          str  = Field(..., env="ENV")
    VERSION:      str  = Field(..., env="VERSION")
    DEBUG:        bool = Field(..., env="DEBUG")
    SECRET_KEY:   str  = Field(..., env="SECRET_KEY")
    ALLOWED_ORIGINS: list[str] = Field(..., env="ALLOWED_ORIGINS")

    # ── Ollama AI settings ─────────────────────────────────────────────────
    # Base URL for Ollama server.
    #   Local:  http://localhost:11434   (default — works with `ollama serve`)
    #   Cloud:  https://your-ollama-server.com
    OLLAMA_BASE_URL: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")

    # Model to use. Must be pulled first: `ollama pull <model>`
    # Recommended models (choose one based on your hardware/cloud tier):
    #   llama3.1        — best quality, needs ~8 GB VRAM / RAM
    #   mistral         — fast, good quality, needs ~5 GB
    #   gemma2          — Google model, strong on structured output
    #   qwen2.5         — excellent JSON adherence
    #   phi3            — lightweight, good for CPU-only servers
    OLLAMA_MODEL: str = Field("llama3.1", env="OLLAMA_MODEL")

    # Request timeout in seconds (increase for large models or slow servers)
    OLLAMA_TIMEOUT: int = Field(120, env="OLLAMA_TIMEOUT")

    # Retry attempts on transient network errors
    OLLAMA_MAX_RETRIES: int = Field(3, env="OLLAMA_MAX_RETRIES")

    # Optional API key — set if your cloud Ollama instance requires auth
    OLLAMA_API_KEY: Optional[str] = Field(None, env="OLLAMA_API_KEY")

    # ── Hugging Face Token (speaker diarization) ───────────────────────────
    HF_TOKEN: str = Field(..., env="HF_TOKEN")

    # ── Behavioral aggregator weights ──────────────────────────────────────
    W_COMMUNICATION:  float = Field(..., env="W_COMMUNICATION")
    W_ENGAGEMENT:     float = Field(..., env="W_ENGAGEMENT")
    W_CONFIDENCE:     float = Field(..., env="W_CONFIDENCE")
    W_PROFESSIONALISM:float = Field(..., env="W_PROFESSIONALISM")

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
