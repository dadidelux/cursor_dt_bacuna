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

# Class labels for coconut disease classifier
coconut_disease_class_names = ['beetles', 'leaf_miner', 'leaf_spot', 'white_flies']

# Class labels for intercropping classifier (alphabetical order of subfolders)
intercropping_class_names = [
    'Cacao', 'Caimito', 'Guava', 'Guyabano', 'Jack Fruit', 'Kabugaw',
    'Kalamansi', 'Mahugani', 'Mango', 'Narra Tree', 'Paper Tree', 'Rambutan', 'Santol'
]

MODEL_INTERCROPPING = 'intercropping_classifier_v2_phase_3.h5'
MODEL_DISEASE = 'best_model_phase_3.h5'
IMG_SIZE = (224, 224)

# Initialize FastAPI app
app = FastAPI(
    title="Coconut Intercropping & Disease Classifier API",
    description="API for classifying coconut diseases and intercropping plants",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction type enum
class PredictionType(str, Enum):
    DISEASE = "disease"
    INTERCROPPING = "intercropping"
    BOTH = "both"

# Response models
class Prediction(BaseModel):
    class_name: str
    confidence: float

class ClassificationResponse(BaseModel):
    disease_predictions: Optional[List[Prediction]] = None
    intercropping_predictions: Optional[List[Prediction]] = None
    most_likely_disease: Optional[str] = None
    most_likely_intercropping: Optional[str] = None

class SimilarImage(BaseModel):
    path: str
    score: float
    leaf_type: str

class VisualSearchResponse(BaseModel):
    similar_images: List[SimilarImage]

# Load models at startup
intercropping_model = None
disease_model = None
clip_model = None
clip_preprocess = None
device = None
faiss_index = None
image_paths = None

@app.on_event("startup")
async def load_models():
    global intercropping_model, disease_model, clip_model, clip_preprocess, device, faiss_index, image_paths
    
    try:
        # Load ML models
        intercropping_model = load_model(MODEL_INTERCROPPING)
        disease_model = load_model(MODEL_DISEASE)
        
        # Load CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # Load FAISS index
        faiss_index = faiss.read_index("intercropping_clip.index")
        image_paths = np.load("intercropping_image_paths.npy")
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load models")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for ML models"""
    img = image.resize(IMG_SIZE)
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

def get_clip_embedding(image: Image.Image) -> np.ndarray:
    """Get CLIP embedding for an image"""
    image = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype('float32')

def find_similar_images(query_image: Image.Image, top_k: int = 5) -> Tuple[List[str], List[float]]:
    """Find similar images using CLIP and FAISS"""
    query_emb = get_clip_embedding(query_image)
    D, I = faiss_index.search(query_emb, top_k)
    similar_paths = [image_paths[i] for i in I[0]]
    return similar_paths, D[0]

@app.post("/predict/", response_model=ClassificationResponse)
async def predict_image(
    file: UploadFile = File(...),
    prediction_type: PredictionType = PredictionType.BOTH
):
    """
    Predict disease and/or intercropping plants from an image
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        response = ClassificationResponse()
        
        # Make predictions based on type
        if prediction_type in [PredictionType.DISEASE, PredictionType.BOTH]:
            disease_preds = predict_single_label(disease_model, image, coconut_disease_class_names)
            response.disease_predictions = [
                Prediction(class_name=name.replace('_', ' ').title(), confidence=conf)
                for name, conf in disease_preds
            ]
            response.most_likely_disease = disease_preds[0][0].replace('_', ' ').title()
            
        if prediction_type in [PredictionType.INTERCROPPING, PredictionType.BOTH]:
            inter_preds = predict_single_label(intercropping_model, image, intercropping_class_names)
            response.intercropping_predictions = [
                Prediction(class_name=name, confidence=conf)
                for name, conf in inter_preds[:5]  # Top 5 predictions
            ]
            response.most_likely_intercropping = inter_preds[0][0]
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visual-search/", response_model=VisualSearchResponse)
async def visual_search(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Find visually similar images using CLIP and FAISS
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Find similar images
        similar_paths, scores = find_similar_images(image, top_k)
        
        # Extract leaf types from paths
        similar_images = []
        for path, score in zip(similar_paths, scores):
            leaf_type = path.split(os.sep)[-2]  # Extract class from path
            similar_images.append(SimilarImage(
                path=path,
                score=float(score),
                leaf_type=leaf_type
            ))
            
        return VisualSearchResponse(similar_images=similar_images)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True) 