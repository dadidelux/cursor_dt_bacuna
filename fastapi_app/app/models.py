from enum import Enum
from pydantic import BaseModel
from typing import List, Optional

class PredictionType(str, Enum):
    DISEASE = "disease"

class Prediction(BaseModel):
    class_name: str
    confidence: float

class VisualMatch(BaseModel):
    leaf_type: str
    similarity_score: float
    image_path: str

class VisualAnalysis(BaseModel):
    matches: List[VisualMatch]
    predicted_class: str
    confidence: float
    analysis_summary: str

class ClassificationResponse(BaseModel):
    disease_predictions: Optional[List[Prediction]] = None
    visual_analysis: Optional[VisualAnalysis] = None
    most_likely_disease: Optional[str] = None

class SimilarImage(BaseModel):
    path: str
    score: float
    leaf_type: str

class VisualSearchResponse(BaseModel):
    similar_images: List[SimilarImage] 