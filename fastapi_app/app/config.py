from pydantic_settings import BaseSettings
from typing import List, Tuple
import os

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Coconut Intercropping & Disease Classifier API"
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # Feature Flags
    ENABLE_DISEASE: bool = os.getenv("ENABLE_DISEASE", "true").lower() == "true"
    ENABLE_INTERCROPPING: bool = os.getenv("ENABLE_INTERCROPPING", "true").lower() == "true"
    
    # Model Paths
    MODEL_BASE_PATH: str = os.getenv("MODEL_BASE_PATH", "models")
    MODEL_INTERCROPPING_PATH: str = os.path.join("models", "intercropping_classifier_transfer_learning_v2.h5")
    MODEL_DISEASE_PATH: str = os.path.join("models", "best_model_phase_3.h5")
    FAISS_INDEX_PATH: str = os.path.join("models", "intercropping_clip.index")
    IMAGE_PATHS_FILE: str = os.path.join("models", "intercropping_image_paths.npy")
    
    # Image Settings
    IMG_SIZE: Tuple[int, int] = (224, 224)
    
    # Class Names
    COCONUT_DISEASE_CLASS_NAMES: List[str] = [
        'beetles', 'leaf_miner', 'leaf_spot', 'white_flies'
    ]
    
    INTERCROPPING_CLASS_NAMES: List[str] = [
        "Cacao", "Caimito", "Guava", "Guyabano", "JackFruit", "Kabugaw",
        "Kalamansi", "Mahugani", "Mango", "Nara", "Paper", "Rambutan", "Sampalok", "Santol"
    ]
    
    class Config:
        case_sensitive = True 