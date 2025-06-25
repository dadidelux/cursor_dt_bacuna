import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import io
import faiss
import torch
import clip
import os

st.set_page_config(
    page_title="Coconut Intercropping & Disease Classifier",
    page_icon="🌴",
    layout="wide"
)

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

# Preprocess image for both models
def preprocess_image(image):
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict for a single-label classifier
def predict_single_label(model, image, class_names):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    class_predictions = []
    for i, pred in enumerate(predictions[0]):
        class_predictions.append((class_names[i], pred * 100))
    class_predictions.sort(key=lambda x: x[1], reverse=True)
    return class_predictions

# 1. Load CLIP model and preprocessing
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# 2. Load FAISS index and image paths
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("intercropping_clip.index")
    image_paths = np.load("intercropping_image_paths.npy")
    return index, image_paths

model, preprocess, device = load_clip_model()
index, image_paths = load_faiss_index()

def get_clip_embedding(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype('float32')

def find_similar_images(query_image, top_k=5):
    query_emb = get_clip_embedding(query_image)
    D, I = index.search(query_emb, top_k)
    similar_paths = [image_paths[i] for i in I[0]]
    return similar_paths, D[0]

# Streamlit app
def main():
    st.title("🌴 Coconut Intercropping & Disease Classifier")
    st.markdown("""
    Upload an image of a coconut tree. The app will identify possible intercropping plants and/or coconut disease status.
    """)

    st.sidebar.header("About")
    st.sidebar.info("""
    **Intercropping classes:**
    - Cacao, Caimito, Guava, Guyabano, Jack Fruit, Kabugaw, Kalamansi, Mahugani, Mango, Narra Tree, Paper Tree, Rambutan, Santol

    **Coconut disease classes:**
    - Beetles, Leaf Miner, Leaf Spot, White Flies
    """)

    # User selects which classifier(s) to use
    option = st.selectbox(
        "Which prediction do you want to run?",
        ("Coconut Disease Only", "Intercropping Only", "Both")
    )

    @st.cache_resource
    def load_intercropping_model():
        return load_model(MODEL_INTERCROPPING)

    @st.cache_resource
    def load_disease_model():
        return load_model(MODEL_DISEASE)

    # Only load models if needed
    intercropping_model = None
    disease_model = None
    try:
        if option in ("Intercropping Only", "Both"):
            intercropping_model = load_intercropping_model()
        if option in ("Coconut Disease Only", "Both"):
            disease_model = load_disease_model()
        st.success("✅ Model(s) loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return

    tab1, tab2 = st.tabs(["Classification", "Visual Search"])

    with tab1:
        st.markdown("### Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Uploaded Image")
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                with col2:
                    st.markdown("### Prediction Results")
                    if option in ("Intercropping Only", "Both"):
                        st.markdown("#### Intercropping Plant Prediction")
                        inter_predictions = predict_single_label(intercropping_model, image, intercropping_class_names)
                        for class_name, confidence in inter_predictions[:5]:
                            color = f"rgba(0, {min(confidence * 2.55, 255)}, 0, 0.2)"
                            st.markdown(f"""
                            <div style=\"padding: 10px; border-radius: 5px; background-color: {color}; margin-bottom: 6px;\">
                                <b>{class_name}:</b> {confidence:.2f}%
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown(f"**Most Likely Intercropping Plant:** <span style='font-size:1.2em'><b>{inter_predictions[0][0]}</b></span>", unsafe_allow_html=True)
                        st.markdown("---")
                    if option in ("Coconut Disease Only", "Both"):
                        st.markdown("#### Coconut Disease Prediction")
                        disease_predictions = predict_single_label(disease_model, image, coconut_disease_class_names)
                        for class_name, confidence in disease_predictions:
                            color = f"rgba(0, {min(confidence * 2.55, 255)}, 0, 0.2)"
                            st.markdown(f"""
                            <div style=\"padding: 10px; border-radius: 5px; background-color: {color}; margin-bottom: 6px;\">
                                <b>{class_name.replace('_', ' ').title()}:</b> {confidence:.2f}%
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown(f"**Most Likely Disease:** <span style='font-size:1.2em'><b>{disease_predictions[0][0].replace('_', ' ').title()}</b></span>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    with tab2:
        st.title("Intercropping Leaf Visual Search (CLIP + FAISS)")

        uploaded_file = st.file_uploader("Upload a leaf/crop image for visual search", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Query Image", use_container_width=True)
            st.write("Searching for similar images...")

            similar_images, scores = find_similar_images(image, top_k=5)
            st.write("Top similar images from dataset:")

            for path, score in zip(similar_images, scores):
                st.image(path, caption=f"Score: {score:.3f}", width=200)
                st.write(f"File: {path}")

            # Optionally, extract the class/leaf type from the path and display it
            # Example: if your path is 'intercropping_classification/train/Guava/img1.jpg'
            leaf_types = [p.split(os.sep)[-2] for p in similar_images]
            st.write("Predicted leaf types (by similarity):", leaf_types)

if __name__ == "__main__":
    main() 