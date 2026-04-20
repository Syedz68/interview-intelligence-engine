from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = Field(..., env="PROJECT_NAME")
    ENV: str = Field(..., env="ENV")
    VERSION: str = Field(..., env="VERSION")
    DEBUG: bool = Field(..., env="DEBUG")
    SECRET_KEY: str = Field(..., env="SECRET_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()