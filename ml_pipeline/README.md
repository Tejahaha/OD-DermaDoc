# Traditional ML Pipeline for Skin Lesion Classification

A modular, beginner-friendly machine learning pipeline for classifying skin lesion images.

## 📁 File Structure

```
ml_pipeline/
├── __init__.py              # Package initialization
├── config.py                # Configuration settings and paths
├── step1_load_data.py       # Load images and labels
├── step2_extract_features.py # Feature extraction (color, texture, shape)
├── step3_train_models.py    # Train ML classifiers
├── step4_visualize.py       # Visualize results
├── step5_inference.py       # Predict on new images
├── run_pipeline.py          # Main pipeline runner
└── results/                 # Output directory (created automatically)
```

## 🚀 Quick Start

### Run the complete pipeline:

```bash
cd ml_pipeline
python run_pipeline.py
```

### Or run individual steps:

```python
# Step 1: Load Data
from step1_load_data import load_images_with_labels
images, labels, paths = load_images_with_labels("path/to/train")

# Step 2: Extract Features
from step2_extract_features import extract_features_from_dataset
X, y, feature_names = extract_features_from_dataset(images, labels)

# Step 3: Train Models
from step3_train_models import train_all_models
results, scaler = train_all_models(X_train, X_val, y_train, y_val)

# Step 4: Visualize
from step4_visualize import generate_full_report
generate_full_report(results, best_name, feature_names, y_val)

# Step 5: Inference
from step5_inference import predict_single_image
prediction, confidence = predict_single_image("path/to/image.jpg")
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Dataset paths**: Where your images are located
- **Training parameters**: Number of trees, learning rate, etc.
- **Sample limits**: For faster testing

## 📊 Pipeline Steps

### Step 1: Load Data
- Loads images from `images/` folder
- Parses class labels from `labels/` folder
- Visualizes sample images and class distribution

### Step 2: Extract Features
- **Color Features** (36): RGB/HSV statistics, color histograms
- **Texture Features** (10): GLCM-based (contrast, homogeneity, etc.)
- **Shape Features** (7): Edge density, circularity, area
- **Statistical Features** (7): Entropy, skewness, kurtosis

Total: **60 features per image**

### Step 3: Train Models
Trains 5 different classifiers:
- 🌲 Random Forest
- 📈 Support Vector Machine (SVM)
- 🎯 K-Nearest Neighbors (KNN)
- 📉 Gradient Boosting
- 🧠 Neural Network (MLP)

### Step 4: Visualize
Creates charts for:
- Model accuracy comparison
- Confusion matrix
- Feature importance

### Step 5: Inference
- Predict on new images
- Get class probabilities
- Visualize predictions

## 📦 Requirements

```bash
pip install numpy opencv-python matplotlib scikit-learn scikit-image seaborn joblib
```

## 🆚 Traditional ML vs Deep Learning

| Aspect | Traditional ML (This) | Deep Learning (YOLOv8) |
|--------|----------------------|------------------------|
| Features | Manual design | Learned automatically |
| Data needed | Less (~hundreds) | More (~thousands) |
| Training time | Minutes | Hours |
| GPU required | No | Yes (recommended) |
| Interpretability | High | Low |
