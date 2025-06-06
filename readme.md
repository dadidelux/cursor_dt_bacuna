# Coconut Pest Classification System

A deep learning system for classifying common coconut pests using EfficientNetV2B0. The system can identify four types of pests with high accuracy.

## Features

- 🔍 Identifies 4 types of coconut pests:
  - Beetles
  - Beal Miner
  - Leaf Spot
  - White Flies
- 🎯 High accuracy (100% on validation set)
- 🖥️ Multiple interfaces:
  - Command-line tool
  - Web interface (Streamlit)
- 🚀 Fast inference
- 📊 Confidence scores for predictions

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)

### Basic Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd coconut-pest-classifier
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### GPU Support (Optional)

For GPU acceleration, you need:
1. NVIDIA GPU with CUDA support
2. CUDA Toolkit 11.8
3. cuDNN 8.6 or later

Installation steps:
1. Download and install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Download and install [cuDNN v8.6](https://developer.nvidia.com/cudnn) (requires NVIDIA account)
3. Add CUDA paths to system environment variables

## Usage

### Command Line Interface

1. Test a single image:
```bash
python test_prediction.py
```
- Follow the prompts to select an image
- View prediction results and confidence scores

### Web Interface

1. Start the Streamlit app:
```bash
python -m streamlit run app.py
```
2. Open your browser at `http://localhost:8501`
3. Upload an image using the interface
4. View predictions and confidence scores

## Project Structure

```
coconut-pest-classifier/
├── app.py                     # Streamlit web interface
├── test_prediction.py         # Command-line testing tool
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── readme_finetune.md        # Model development details
├── best_model_phase_3.h5     # Trained model
└── dataset/                  # Dataset directory
    ├── training/             # Training images
    └── validation/          # Validation images
```

## Model Performance

- Training Accuracy: 99.48%
- Validation Accuracy: 100%
- Loss: 0.0114

For detailed information about the model architecture and training process, see [readme_finetune.md](readme_finetune.md).

## Example Usage

### Command Line Interface
```bash
> python test_prediction.py

Available validation images per class:
Beetles: 23 images
Beal Miner: 28 images
Leaf Spot: 34 images
White Flies: 32 images

Options:
1. Test random image
2. Test specific class
3. Quit
```

### Web Interface
![Streamlit Interface](screenshots/streamlit_interface.png)
1. Upload an image using the file uploader
2. View the image preview
3. See prediction results with confidence bars
4. Check detailed model information in the sidebar

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   - Verify CUDA installation: `nvidia-smi`
   - Check TensorFlow GPU support: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
   - Ensure CUDA version matches TensorFlow requirements

2. **Import Errors**
   - Verify virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

3. **Model Loading Error**
   - Check if model file exists in correct location
   - Verify TensorFlow version compatibility

### Getting Help

For more information:
- Check the [model fine-tuning documentation](readme_finetune.md)
- Submit an issue on GitHub
- Contact the maintainers

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset contributors
- EfficientNet team for the base model
- TensorFlow and Keras teams
- Streamlit team for the web framework