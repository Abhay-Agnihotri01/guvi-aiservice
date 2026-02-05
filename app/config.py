# Set environment variables BEFORE any other imports to fix Windows symlink issues
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "EchoTruth AI Service"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model paths
    models_dir: str = "app/models"
    
    # Supported languages
    supported_languages: list = ["tamil", "english", "hindi", "malayalam", "telugu"]
    
    # Model weights for ensemble
    wav2vec_weight: float = 0.7
    acoustic_weight: float = 0.2
    spectral_weight: float = 0.1
    gemini_weight: float = 0.0
    
    # Detection threshold
    ai_threshold: float = 0.5
    
    # API Keys
    google_api_key: str = ""
    
    class Config:
        env_file = "../.env"
        extra = "ignore"

settings = Settings()
