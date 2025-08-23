# Coconut Intercropping & Disease Classifier API

A FastAPI-based REST API for classifying coconut diseases and intercropping plants using deep learning models.

## Features

- 🔍 Disease Classification:
  - Beetles
  - Leaf Miner
  - Leaf Spot
  - White Flies
- 🌱 Intercropping Plant Classification:
  - Multiple plant species including Cacao, Guava, Mango, etc.
- 🖼️ Visual Search using CLIP and FAISS
- 🚀 Fast inference with TensorFlow
- 📊 Confidence scores for predictions
- 🔒 Production-rady with Nginx reverse proxy

## API Endpoints

### 1. Predict Image (`POST /predict/`)
Upload an image to get disease and/or intercropping predictions.

**Parameters:**
- `file`: Image file (jpg, jpeg, png)
- `prediction_type`: Type of prediction ("disease", "intercropping", or "both")

### 2. Visual Search (`POST /visual-search/`)
Find visually similar images using CLIP embeddings.

**Parameters:**
- `file`: Image file (jpg, jpeg, png)
- `top_k`: Number of similar images to return (default: 5)

### 3. Health Check (`GET /health`)
Check if the API is running.

## Setup

### Prerequisites

- Docker and Docker Compose
- CUDA-capable GPU (optional, for faster inference)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd fastapi_app
```

2. Copy required model files to the FastAPI app directory:
```bash
cp ../best_model_phase_3.h5 .
cp ../intercropping_classifier_v2_phase_3.h5 .
cp ../intercropping_clip.index .
cp ../intercropping_image_paths.npy .
```

3. Build and run with Docker Compose:
```bash
docker-compose up --build
```

The API will be available at:
- FastAPI: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Alternative Documentation: http://localhost:8000/redoc

### Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
uvicorn app.main:app --reload
```

## Environment Variables

The following environment variables can be set in docker-compose.yml:

- `MODEL_INTERCROPPING_PATH`: Path to intercropping model file
- `MODEL_DISEASE_PATH`: Path to disease model file
- `FAISS_INDEX_PATH`: Path to FAISS index file
- `IMAGE_PATHS_FILE`: Path to image paths numpy file

## API Documentation

Full API documentation is available at:
- Swagger UI: `/docs`
- ReDoc: `/redoc`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 


## Notes
  Current Results:
  - Phase 1: 56.8% validation accuracy
  - Phase 2: 63.8% validation accuracy
  - Phase 3: 66.7% validation accuracy