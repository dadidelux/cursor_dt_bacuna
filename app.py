import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Class labels
class_names = ['beetles', 'beal_miner', 'leaf_spot', 'white_flies']

def preprocess_image(image):
    """Preprocess a PIL image for prediction."""
    # Resize image
    img = image.resize((224, 224))
    # Convert to array and add batch dimension
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess input (same as training)
    img_array = img_array / 255.0
    return img_array

def predict_image(model, image):
    """Make prediction for a PIL image."""
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get predictions for all classes
    class_predictions = []
    for i, pred in enumerate(predictions[0]):
        class_predictions.append((class_names[i], pred * 100))
    
    # Sort by confidence
    class_predictions.sort(key=lambda x: x[1], reverse=True)
    
    return class_predictions

def main():
    st.set_page_config(
        page_title="Coconut Pest Classifier",
        page_icon="🌴",
        layout="wide"
    )
    
    # Title and description
    st.title("🌴 Coconut Pest Classifier")
    st.markdown("""
    This application uses a deep learning model to classify common coconut pests.
    Upload an image of a coconut pest or disease symptom to get predictions.
    """)
    
    # Sidebar information
    st.sidebar.header("About")
    st.sidebar.info("""
    This model can detect the following pests:
    1. Beetles
    2. Beal Miner
    3. Leaf Spot
    4. White Flies
    
    The model was trained on a dataset of coconut pest images using EfficientNetV2B0 architecture.
    """)
    
    # Load model
    @st.cache_resource
    def load_classifier_model():
        return load_model('best_model_phase_3.h5')
    
    try:
        model = load_classifier_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # File uploader
    st.markdown("### Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Uploaded Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.markdown("### Prediction Results")
                
                # Make prediction
                predictions = predict_image(model, image)
                
                # Display results
                for class_name, confidence in predictions:
                    # Create a color gradient based on confidence
                    color = f"rgba(0, {min(confidence * 2.55, 255)}, 0, 0.2)"
                    
                    # Display each prediction with a progress bar
                    st.markdown(f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: {color}; margin-bottom: 10px;">
                        <b>{class_name.replace('_', ' ').title()}:</b> {confidence:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display most likely class
                st.markdown("---")
                st.markdown(f"""
                ### Most Likely:
                # {predictions[0][0].replace('_', ' ').title()}
                """)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 