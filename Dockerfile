# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and application files
COPY app.py .
COPY best_model_phase_3.h5 .

# If you have a dataset directory that's needed for validation images, uncomment and modify this line
# COPY dataset/ ./dataset/

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Create a new user with limited privileges
RUN useradd -m -r streamlit
RUN chown -R streamlit:streamlit /app
USER streamlit

# Run streamlit when the container launches
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"] 