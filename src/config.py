from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    TEMP_FILE_PATH: str = "temp.pdf"
    CHROMA_DB_PATH: str = "db/chroma_db"
    CHROMA_CLIP_DB_PATH: str = "db/chroma_clip_db"
    # Add other configurations here in the future

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
