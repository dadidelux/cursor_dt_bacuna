# Use TensorFlow GPU image for CUDA support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and additional Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install additional ML packages
RUN pip install --no-cache-dir \
    matplotlib \
    scikit-learn \
    Pillow \
    tqdm

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training and application files
COPY intercropping_classifier.py .
COPY predict_with_tta.py .
COPY app.py .
COPY fastapi_app.py .

# Copy dataset and models if they exist
COPY datasets-source/ ./datasets-source/
COPY intercropping_classification/ ./intercropping_classification/

# If you have a dataset directory that's needed for validation images, uncomment and modify this line
# COPY dataset/ ./dataset/

# Make ports available to the world outside this container
EXPOSE 8000 8501

# Create a new user with limited privileges
RUN useradd -m -r appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set environment variables for CUDA
ENV CUDA_VISIBLE_DEVICES=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Default command - train the model
CMD ["python", "intercropping_classifier.py"] 