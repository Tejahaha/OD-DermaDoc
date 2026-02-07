"""
=============================================================================
CONFIGURATION SETTINGS
=============================================================================

This module contains all configuration settings for the ML pipeline.
Edit the paths and parameters here to customize the pipeline behavior.

Usage:
    from config import DATASET_PATH, TRAIN_PATH, OUTPUT_DIR
"""

import os

# ============================================================================
# DATASET PATHS
# ============================================================================

# Set to True to use the balanced dataset after running step0_prepare_dataset.py
USE_BALANCED_DATASET = True

# Original dataset path
ORIGINAL_DATASET_PATH = r"c:\Users\TEJA\PycharmProjects\DermaDoc\skin-lesion-segmentation-classification"

# Balanced dataset path (created by step0_prepare_dataset.py)
BALANCED_DATASET_PATH = r"c:\Users\TEJA\PycharmProjects\DermaDoc\balanced_dataset"

# Active dataset path (switches based on USE_BALANCED_DATASET flag)
DATASET_PATH = BALANCED_DATASET_PATH if USE_BALANCED_DATASET else ORIGINAL_DATASET_PATH

# Paths to each split (train/validation/test)
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALID_PATH = os.path.join(DATASET_PATH, "valid")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# Output directory for results and saved models
OUTPUT_DIR = r"c:\Users\TEJA\PycharmProjects\DermaDoc\ml_pipeline\results"


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

# Class ID to Description mapping
CLASS_DESCRIPTIONS = {
    0: "Non-cancerous, often scaly skin lesions. Common and usually harmless.",
    1: "Regular moles formed by pigment-producing cells. Typically benign.",
    2: "Firm, small nodules under the skin caused by minor trauma. Non-cancerous.",
    3: "A serious and potentially life-threatening skin cancer. Early detection is critical.",
    4: "Blood vessel-related marks like angiomas. Usually red or purple.",
    5: "The most common skin cancer. Slow-growing and rarely metastasizes.",
    6: "Pre-cancerous lesions that may evolve into squamous cell carcinoma.",
}

# Class severity levels (for visualization)
# 0 = Benign, 1 = Pre-cancerous, 2 = Malignant
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
# IMAGE PROCESSING SETTINGS
# ============================================================================

# Image size for feature extraction (images will be resized to this)
IMAGE_SIZE = (256, 256)

# Supported image file extensions
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]


# ============================================================================
# FEATURE EXTRACTION SETTINGS
# ============================================================================

# GLCM (Gray Level Co-occurrence Matrix) settings
GLCM_DISTANCES = [1, 2]                           # Pixel distances
GLCM_ANGLES = [0, 0.785, 1.571, 2.356]           # 0°, 45°, 90°, 135° in radians
GLCM_LEVELS = 16                                   # Number of gray levels

# Color histogram settings
COLOR_HISTOGRAM_BINS = 8                           # Bins per channel


# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================

# Random Forest parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# SVM parameters
SVM_C = 10
SVM_KERNEL = 'rbf'

# KNN parameters
KNN_NEIGHBORS = 5

# Gradient Boosting parameters
GB_N_ESTIMATORS = 100
GB_MAX_DEPTH = 5
GB_LEARNING_RATE = 0.1

# Neural Network (MLP) parameters
MLP_HIDDEN_LAYERS = (100, 50)
MLP_MAX_ITER = 500


# ============================================================================
# DATA LOADING SETTINGS
# ============================================================================

# Maximum samples to load (None = load all, set a number for faster testing)
MAX_TRAIN_SAMPLES = None  # Set to None for full dataset
MAX_VALID_SAMPLES = None  # Set to None for full dataset
MAX_TEST_SAMPLES = None   # Set to None for full dataset


# ============================================================================
# RANDOM SEED (for reproducibility)
# ============================================================================

RANDOM_SEED = 42


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


# Create output directory on import
ensure_output_dir()
