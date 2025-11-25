# Sign2Speech

A complete Sign Language to Text and Speech translation system, built from scratch with custom neural networks. This project recognizes American Sign Language (ASL) hand gestures from webcam input and converts them to both text and spoken audio.

## 🎯 Project Overview

Sign2Speech is an end-to-end system that:
- **Captures** sign language gestures from a webcam in real-time
- **Recognizes** signs using a custom CNN trained from scratch (no pre-trained models)
- **Displays** the predicted sign as text
- **Speaks** the prediction using text-to-speech

### Key Features

- ✅ **Custom Model**: CNN architecture built and trained from scratch
- ✅ **No Transfer Learning**: All weights randomly initialized and trained on dataset
- ✅ **Real-time Inference**: Live webcam feed with instant predictions
- ✅ **Text-to-Speech**: Converts predictions to spoken audio
- ✅ **Easy to Use**: Simple Streamlit web interface
- ✅ **Well Documented**: Clean code structure with comprehensive documentation

## 📋 Requirements

- Python 3.8+
- Webcam (for real-time demo)
- CUDA-capable GPU (optional, for faster training)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

The project uses the **ASL Alphabet Dataset** from Kaggle:

**Option A: Using Kaggle API (Recommended)**

1. Install Kaggle CLI and authenticate:
   ```bash
   pip install kaggle
   # Place your kaggle.json credentials file in ~/.kaggle/ or current directory
   ```

2. Run the download script:
   ```bash
   python scripts/download_data.py
   ```

**Option B: Manual Download**

1. Go to [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
2. Download the dataset
3. Extract to `data/raw/`
4. You should have: `data/raw/asl_alphabet_train/` with class folders (A, B, C, ..., Z, del, nothing, space)

**Alternative Dataset**: You can also use [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) - modify `src/datasets.py` accordingly.

### 3. Train the Model

Train a custom CNN from scratch:

```bash
python -m src.train --data-root data/raw --epochs 20 --batch-size 32 --lr 0.001
```

**Training Arguments:**
- `--data-root`: Dataset root directory (default: `data/raw`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--image-size`: Input image size (default: 64)
- `--model-type`: Model type - `simple` or `smaller` (default: `simple`)
- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints`)
- `--val-split`: Validation split ratio (default: 0.2)
- `--no-augment`: Disable data augmentation
- `--seed`: Random seed for reproducibility (default: 42)

**Example with GPU:**
```bash
python -m src.train --data-root data/raw --epochs 30 --batch-size 64 --image-size 64
```

The training script will:
- Create train/validation splits
- Train the model with progress bars
- Save checkpoints to `checkpoints/`
- Save the best model based on validation accuracy
- Save training history and class mappings

### 4. Evaluate the Model

Evaluate a trained model:

```bash
python -m src.evaluate --checkpoint checkpoints/best.pth --data-root data/raw
```

This will:
- Compute accuracy and per-class metrics
- Generate a confusion matrix
- Save results to `evaluation_results/`

### 5. Run the Real-time Demo

Launch the Streamlit web app:

```bash
streamlit run app/app.py
```

Or:

```bash
python -m streamlit run app/app.py
```

The app will open in your browser. Click "Start Camera" to begin recognizing signs in real-time!

## 📁 Project Structure

```
Sign2Speech/
├── data/
│   ├── raw/              # Original downloaded dataset
│   └── processed/        # Processed images/keypoints (if needed)
├── notebooks/            # Optional: Jupyter notebooks for experimentation
├── src/
│   ├── __init__.py
│   ├── datasets.py       # Dataset loading & transforms
│   ├── model.py          # Custom CNN model definitions
│   ├── train.py          # Training loop, validation, checkpointing
│   ├── evaluate.py       # Evaluation + metrics
│   ├── infer.py          # Single-sample inference helpers
│   ├── keypoints.py      # MediaPipe hand keypoint extraction
│   └── tts.py            # Text-to-speech helper
├── app/
│   └── app.py            # Streamlit real-time demo
├── scripts/
│   └── download_data.py  # Dataset download script
├── checkpoints/           # Saved model checkpoints (created during training)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🏗️ Architecture

### Model Architecture

The project uses a **custom CNN built from scratch**:

**SimpleCNN** (default):
- 4 Convolutional blocks (Conv2d → ReLU → MaxPool)
- 3 Fully Connected layers with dropout
- All weights randomly initialized (Xavier/Kaiming)
- Input: 64x64 RGB images
- Output: 26 classes (ASL alphabet A-Z)

**SmallerCNN** (alternative):
- 3 Convolutional blocks
- 2 Fully Connected layers
- Faster training, slightly lower accuracy

### Training Pipeline

1. **Data Loading**: PyTorch Dataset with train/val splits
2. **Augmentation**: Random rotation, color jitter, horizontal flip
3. **Training**: Adam optimizer with learning rate scheduling
4. **Validation**: Monitor validation accuracy, save best model
5. **Checkpointing**: Save model state, optimizer state, and metrics

### Inference Pipeline

1. **Webcam Capture**: OpenCV captures frames
2. **Preprocessing**: Resize, normalize (matching training transforms)
3. **Prediction**: Forward pass through trained model
4. **Post-processing**: Softmax for probabilities, argmax for class
5. **TTS**: Convert predicted class to speech

## 🔧 Usage Examples

### Single Image Inference

```python
from src.infer import SignLanguageInference

# Load model
engine = SignLanguageInference("checkpoints/best.pth", image_size=64)

# Predict from image file
predicted_class, confidence, all_probs = engine.predict_file("test_image.jpg")
print(f"Predicted: {predicted_class} (confidence: {confidence:.2%})")
```

### Command-line Inference

```bash
python -m src.infer --checkpoint checkpoints/best.pth --image test_image.jpg
```

### Text-to-Speech

```python
from src.tts import TextToSpeech

# Create TTS engine
tts = TextToSpeech("pyttsx3")  # or "gtts"

# Speak text
tts.speak("Hello, this is a test")
```

## 📊 Model Performance

After training on the ASL Alphabet dataset:
- **Training Accuracy**: ~95-98% (varies by epochs)
- **Validation Accuracy**: ~90-95% (varies by epochs)
- **Model Size**: ~2-5 MB (depending on architecture)

*Note: Actual performance depends on training hyperparameters, dataset quality, and number of epochs.*

## 🎓 Training Tips

1. **Start Small**: Use `--model-type smaller` for faster iteration
2. **Data Augmentation**: Keep it enabled (default) for better generalization
3. **Learning Rate**: Start with 0.001, reduce if loss plateaus
4. **Batch Size**: Increase if you have GPU memory (faster training)
5. **Epochs**: Train for at least 20 epochs, monitor validation accuracy
6. **Early Stopping**: Manually stop if validation accuracy plateaus

## 🐛 Troubleshooting

### Camera not working
- Check camera permissions
- Try a different camera index in `app/app.py` (change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)
- Ensure no other application is using the camera

### TTS not working
- **pyttsx3**: Install system TTS drivers (varies by OS)
- **gTTS**: Requires internet connection
- Try switching TTS engines in the Streamlit sidebar

### Out of memory during training
- Reduce `--batch-size` (e.g., `--batch-size 16`)
- Use `--model-type smaller`
- Reduce `--image-size` (e.g., `--image-size 48`)

### Low accuracy
- Train for more epochs
- Check dataset quality and class balance
- Adjust learning rate
- Try different model architectures

## 🔬 Customization

### Adding New Classes

1. Add new class folders to `data/raw/asl_alphabet_train/`
2. Retrain the model (it will automatically detect new classes)
3. The model will output the correct number of classes

### Using Different Datasets

Modify `src/datasets.py` to support different dataset structures. The `ASLDataset` class expects:
- Root directory with class folders
- Each class folder contains images

### Model Architecture Changes

Edit `src/model.py` to:
- Add more layers
- Change filter sizes
- Modify activation functions
- Adjust dropout rates

## 📝 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **Dataset**: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) by Grassknoted
- **Libraries**: PyTorch, Streamlit, OpenCV, MediaPipe, pyttsx3, gTTS

## 📧 Contact

For questions, issues, or contributions, please open an issue on the repository.

ndaquan2007@gmail.com

**Built with ❤️ from scratch - no pre-trained models, no transfer learning, just pure custom neural networks!**

