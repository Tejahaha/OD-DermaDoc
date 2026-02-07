"""
=============================================================================
DL PIPELINE CONFIGURATION
=============================================================================

This module contains all configuration settings for the YOLOv8 deep learning
pipeline for skin lesion segmentation.

Usage:
    from config import DATASET_PATH, TRAIN_PATH, MODEL_SIZE
"""

import os
import torch

# ============================================================================
# DATASET PATHS
# ============================================================================

# Base path to the segmentation dataset (YOLO format)
DATASET_PATH = r"c:\Users\TEJA\PycharmProjects\DermaDoc\skin-lesion-segmentation-classification"

# Paths to each split
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALID_PATH = os.path.join(DATASET_PATH, "valid")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# Output directory for results, models, and visualizations
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# ============================================================================
# CLASS DEFINITIONS (Skin Lesion Types)
# ============================================================================

# Class ID to Code mapping
CLASS_CODES = {
    0: "BKL",    # Benign Keratosis
    1: "NV",     # Melanocytic Nevi
    2: "DF",     # Dermatofibroma
    3: "MEL",    # Melanoma
    4: "VASC",   # Vascular Lesion
    5: "BCC",    # Basal Cell Carcinoma
    6: "AKIEC",  # Actinic Keratoses / Intraepithelial Carcinoma
}

# Class ID to Full Name mapping
CLASS_NAMES = {
    0: "Benign Keratosis",
    1: "Melanocytic Nevi",
    2: "Dermatofibroma",
    3: "Melanoma",
    4: "Vascular Lesion",
    5: "Basal Cell Carcinoma",
    6: "Actinic Keratoses",
}

# Class severity levels (0 = Benign, 1 = Pre-cancerous, 2 = Malignant)
CLASS_SEVERITY = {
    0: 0,  # BKL - Benign
    1: 0,  # NV - Benign
    2: 0,  # DF - Benign
    3: 2,  # MEL - Malignant (CRITICAL)
    4: 0,  # VASC - Benign
    5: 2,  # BCC - Malignant
    6: 1,  # AKIEC - Pre-cancerous
}

# Number of classes
NUM_CLASSES = len(CLASS_CODES)


# ============================================================================
# YOLOV8 MODEL SETTINGS
# ============================================================================

# Model size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
# Recommended for RTX 5050 8GB: 'n' or 's'
MODEL_SIZE = 'n'

# Model variant (segmentation)
MODEL_NAME = f"yolov8{MODEL_SIZE}-seg.pt"


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Number of training epochs
# Quick test: 2-5, Normal: 50-100, Full training: 100-300
EPOCHS = 50

# Batch size (reduce if GPU memory errors occur)
# RTX 5050 8GB: recommended 8-16
BATCH_SIZE = 8

# Input image size (must be divisible by 32)
IMAGE_SIZE = 640

# Learning rate (YOLO default is usually good)
LEARNING_RATE = 0.01

# Weight decay for regularization
WEIGHT_DECAY = 0.0005

# Patience for early stopping (epochs without improvement)
PATIENCE = 10

# Workers for data loading
WORKERS = 4


# ============================================================================
# GPU SETTINGS
# ============================================================================

# Device selection
def get_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = 'cpu'
        print("⚠️ No GPU detected, using CPU (training will be slow)")
    return device

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Colors for class visualization (BGR format for OpenCV)
CLASS_COLORS = [
    (255, 100, 100),   # Light Blue - BKL
    (100, 255, 100),   # Light Green - NV
    (100, 100, 255),   # Light Red - DF
    (0, 0, 255),       # Red - MEL (danger!)
    (255, 200, 100),   # Cyan - VASC
    (0, 100, 255),     # Orange - BCC (danger!)
    (200, 100, 255),   # Pink - AKIEC
]

# Confidence threshold for inference
CONF_THRESHOLD = 0.25

# IoU threshold for NMS
IOU_THRESHOLD = 0.45


# ============================================================================
# SUPPORTED FILE EXTENSIONS
# ============================================================================

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# ============================================================================
# RANDOM SEED (for reproducibility)
# ============================================================================

RANDOM_SEED = 42


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def get_data_yaml_path():
    """Get path to data.yaml file."""
    return os.path.join(DATASET_PATH, "data.yaml")


def print_config():
    """Print current configuration."""
    print("\n" + "=" * 60)
    print("DL PIPELINE CONFIGURATION")
    print("=" * 60)
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Model: YOLOv8{MODEL_SIZE}-seg")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60 + "\n")


# Create output directory on import
ensure_output_dir()
