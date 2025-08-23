import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import io
import faiss
import torch
import clip
from enum import Enum
from pydantic import BaseModel
from typing import List, Tuple, Optional
import uvicorn
from collections import Counter
import re
from fastapi import APIRouter

from .models import (
    PredictionType,
    Prediction,
    ClassificationResponse,
    VisualMatch,
    VisualAnalysis
)
from .config import Settings

settings = Settings()

# Initialize FastAPI app
app = FastAPI(
    title="Coconut Disease & Visual Plant Classifier API",
    description="API for classifying coconut diseases and plants using visual similarity",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
disease_model = None
clip_model = None
clip_preprocess = None
device = None
faiss_index = None
image_paths = None
intercropping_model = None

# Remove CLIP, FAISS, and visual prediction logic from startup
@app.on_event("startup")
async def load_models():
    global disease_model, intercropping_model
    try:
        if settings.ENABLE_DISEASE:
            disease_model = load_model(settings.MODEL_DISEASE_PATH)
        if settings.ENABLE_INTERCROPPING:
            intercropping_model = load_model(settings.MODEL_INTERCROPPING_PATH)
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load models")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for ML models"""
    img = image.resize(settings.IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_single_label(model: tf.keras.Model, image: Image.Image, class_names: List[str]) -> List[Tuple[str, float]]:
    """Make prediction using a single-label classifier"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    class_predictions = []
    for i, pred in enumerate(predictions[0]):
        class_predictions.append((class_names[i], float(pred * 100)))
    class_predictions.sort(key=lambda x: x[1], reverse=True)
    return class_predictions

def extract_leaf_type(path):
    # Handles both / and \\ as separators and extracts the class name
    match = re.search(r"intercropping_classification[\\/](?:train|validation)[\\/](.*?)[\\/]", path)
    if match:
        return match.group(1)
    else:
        # fallback: just get the parent folder
        return os.path.basename(os.path.dirname(path))

def analyze_visual_matches(similar_paths: List[str], similarity_scores: List[float], top_k: int = 5) -> VisualAnalysis:
    """Analyze visual matches to determine the most likely class"""
    matches = []
    leaf_types = []
    
    # Create matches and collect leaf types
    for path, score in zip(similar_paths[:top_k], similarity_scores[:top_k]):
        leaf_type = extract_leaf_type(path)
        matches.append(VisualMatch(
            leaf_type=leaf_type,
            similarity_score=float(score),
            image_path=path
        ))
        leaf_types.append(leaf_type)
    
    # Count occurrences of each leaf type
    leaf_type_counts = Counter(leaf_types)
    most_common_type = leaf_type_counts.most_common(1)[0]
    
    # Calculate confidence based on majority and similarity scores
    majority_count = most_common_type[1]
    predicted_class = most_common_type[0]
    
    # Weight the confidence by both frequency and similarity scores
    type_scores = {}
    for match in matches:
        if match.leaf_type not in type_scores:
            type_scores[match.leaf_type] = []
        type_scores[match.leaf_type].append(match.similarity_score)
    
    # Calculate average similarity score for the predicted class
    avg_similarity = sum(type_scores[predicted_class]) / len(type_scores[predicted_class])
    confidence = (majority_count / top_k) * avg_similarity * 100
    
    # Create analysis summary
    analysis_summary = (
        f"Found {majority_count} matches for {predicted_class} out of {top_k} similar images. "
        f"Average similarity score for {predicted_class}: {avg_similarity:.2%}. "
        f"Other candidates: {', '.join(f'{type_}({count})' for type_, count in leaf_type_counts.most_common()[1:])}"
    )
    
    return VisualAnalysis(
        matches=matches,
        predicted_class=predicted_class,
        confidence=confidence,
        analysis_summary=analysis_summary
    )

# Remove visual prediction logic from /predict/ endpoint
@app.post("/predict/", response_model=ClassificationResponse)
async def predict_image(
    prediction_type: PredictionType,
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Predict using only the disease classifier.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        response = ClassificationResponse()
        if prediction_type == PredictionType.DISEASE:
            if not disease_model:
                raise HTTPException(status_code=400, detail="Disease prediction is not enabled")
            disease_preds = predict_single_label(disease_model, image, settings.COCONUT_DISEASE_CLASS_NAMES)
            response.disease_predictions = [
                Prediction(class_name=name.replace('_', ' ').title(), confidence=conf)
                for name, conf in disease_preds
            ]
            response.most_likely_disease = disease_preds[0][0].replace('_', ' ').title()
        else:
            raise HTTPException(status_code=400, detail="Only disease prediction is supported.")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/intercropping/")
async def predict_intercropping(file: UploadFile = File(...)):
    """
    Predict the intercropping class using the Keras model.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        preds = predict_single_label(intercropping_model, image, settings.INTERCROPPING_CLASS_NAMES)
        return {
            "top_prediction": {"class_name": preds[0][0], "confidence": preds[0][1]},
            "all_predictions": [
                {"class_name": name, "confidence": conf} for name, conf in preds
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 