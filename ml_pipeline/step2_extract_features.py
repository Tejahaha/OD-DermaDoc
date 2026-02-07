"""
=============================================================================
STEP 2: FEATURE EXTRACTION
=============================================================================

This module extracts meaningful numerical features from images that can be
used by machine learning classifiers.

WHAT ARE FEATURES?
==================
Instead of using raw pixels (millions of numbers), we extract meaningful 
characteristics:

1. COLOR FEATURES     - RGB/HSV statistics and histograms
2. TEXTURE FEATURES   - GLCM-based texture descriptors
3. SHAPE FEATURES     - Edge density, circularity, contour properties
4. STATISTICAL FEATURES - Entropy, skewness, kurtosis

Key Functions:
    - extract_color_features(): Extract color-based features
    - extract_texture_features(): Extract GLCM texture features
    - extract_shape_features(): Extract shape/edge features
    - extract_all_features(): Combine all feature types
    - extract_features_from_dataset(): Process entire dataset

Usage:
    from step2_extract_features import extract_features_from_dataset
    
    X, y, feature_names = extract_features_from_dataset(images, labels)
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# For texture feature extraction
from skimage.feature import graycomatrix, graycoprops

# Import configuration
from config import (
    OUTPUT_DIR, IMAGE_SIZE, RANDOM_SEED,
    GLCM_DISTANCES, GLCM_ANGLES, GLCM_LEVELS,
    COLOR_HISTOGRAM_BINS
)

# Set random seed
np.random.seed(RANDOM_SEED)


# ============================================================================
# COLOR FEATURES
# ============================================================================

def extract_color_features(image):
    """
    Extract color-based features from an image.
    
    WHY COLOR FEATURES?
    -------------------
    Skin lesions often have distinctive colors (dark brown, pink, red, etc.)
    that differ from normal skin. Color statistics capture these differences.
    
    FEATURES EXTRACTED:
    -------------------
    - RGB channel mean and standard deviation (6 features)
    - HSV channel mean and standard deviation (6 features)
    - Color histogram per channel (8 bins × 3 channels = 24 features)
    
    Total: 36 color features
    
    Parameters:
        image (numpy.ndarray): RGB image array
    
    Returns:
        features (numpy.ndarray): 1D array of color features
        feature_names (list): Names of each feature
    """
    features = []
    feature_names = []
    
    # -------------------------------------------------------------------------
    # RGB Statistics
    # -------------------------------------------------------------------------
    channel_names = ['R', 'G', 'B']
    for i, name in enumerate(channel_names):
        channel = image[:, :, i]
        
        # Mean: average color intensity
        mean_val = np.mean(channel)
        features.append(mean_val)
        feature_names.append(f'{name}_mean')
        
        # Standard deviation: color variation
        std_val = np.std(channel)
        features.append(std_val)
        feature_names.append(f'{name}_std')
    
    # -------------------------------------------------------------------------
    # HSV Statistics (Hue, Saturation, Value)
    # -------------------------------------------------------------------------
    # HSV is often better for color analysis:
    # - H (Hue): actual color (red, green, blue, etc.)
    # - S (Saturation): color intensity/purity
    # - V (Value): brightness
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_names = ['H', 'S', 'V']
    
    for i, name in enumerate(hsv_names):
        channel = hsv[:, :, i]
        
        mean_val = np.mean(channel)
        features.append(mean_val)
        feature_names.append(f'{name}_mean')
        
        std_val = np.std(channel)
        features.append(std_val)
        feature_names.append(f'{name}_std')
    
    # -------------------------------------------------------------------------
    # Color Histograms
    # -------------------------------------------------------------------------
    # Histograms capture the distribution of pixel values
    # More robust than just mean/std
    
    for i, name in enumerate(channel_names):
        hist = cv2.calcHist([image], [i], None, [COLOR_HISTOGRAM_BINS], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize to sum to 1
        
        for bin_idx, val in enumerate(hist):
            features.append(val)
            feature_names.append(f'{name}_hist_bin{bin_idx}')
    
    return np.array(features), feature_names


# ============================================================================
# TEXTURE FEATURES (GLCM)
# ============================================================================

def extract_texture_features(image):
    """
    Extract texture features using Gray-Level Co-occurrence Matrix (GLCM).
    
    WHAT IS GLCM?
    -------------
    GLCM captures spatial relationships between pixels. It counts how often
    pairs of gray-level values occur at specific distances and angles.
    
    Example: How often does gray level 100 appear next to gray level 105?
    
    WHY TEXTURE FEATURES?
    ---------------------
    Skin lesions have distinct textures compared to normal skin:
    - Melanomas may have irregular, rough textures
    - Benign lesions often have smoother textures
    
    FEATURES EXTRACTED:
    -------------------
    From GLCM, we compute these properties (mean and std across angles):
    
    - Contrast:      Intensity contrast between neighboring pixels
                     High = large local intensity variations
    
    - Dissimilarity: Similar to contrast, measures local variation
    
    - Homogeneity:   How uniform/smooth the texture is
                     High = similar gray levels close together
    
    - Energy:        Measures texture uniformity
                     High = more uniform, fewer gray level pairs
    
    - Correlation:   Linear dependency of gray levels
                     High = predictable patterns
    
    Total: 10 texture features (5 properties × 2 statistics)
    
    Parameters:
        image (numpy.ndarray): RGB image array
    
    Returns:
        features (numpy.ndarray): 1D array of texture features
        feature_names (list): Names of each feature
    """
    features = []
    feature_names = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize for consistent processing
    gray = cv2.resize(gray, IMAGE_SIZE)
    
    # Quantize gray levels (256 → GLCM_LEVELS)
    # This reduces computation and makes GLCM more robust
    gray = (gray / 256 * GLCM_LEVELS).astype(np.uint8)
    
    # Compute GLCM
    # Multiple angles capture texture in all directions
    glcm = graycomatrix(
        gray,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True
    )
    
    # Extract properties
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    for prop in properties:
        values = graycoprops(glcm, prop)
        
        # Mean across all distances and angles
        mean_val = values.mean()
        features.append(mean_val)
        feature_names.append(f'glcm_{prop}_mean')
        
        # Std captures variation across directions
        std_val = values.std()
        features.append(std_val)
        feature_names.append(f'glcm_{prop}_std')
    
    return np.array(features), feature_names


# ============================================================================
# SHAPE FEATURES
# ============================================================================

def extract_shape_features(image):
    """
    Extract shape-based features using edge detection and contour analysis.
    
    WHY SHAPE FEATURES?
    -------------------
    Malignant lesions often have irregular, asymmetric shapes.
    Benign lesions tend to be more circular and well-defined.
    
    FEATURES EXTRACTED:
    -------------------
    - Edge density:     Ratio of edge pixels to total pixels
    - Normalized area:  Lesion area relative to image size
    - Normalized perimeter: Contour length relative to image dimensions
    - Circularity:      How close to a perfect circle (1.0 = circle)
    - Aspect ratio:     Width/height of bounding box
    - Extent:           Lesion area / bounding box area
    - Solidity:         Lesion area / convex hull area
    
    Total: 7 shape features
    
    Parameters:
        image (numpy.ndarray): RGB image array
    
    Returns:
        features (numpy.ndarray): 1D array of shape features
        feature_names (list): Names of each feature
    """
    features = []
    feature_names = [
        'edge_density',
        'norm_area',
        'norm_perimeter',
        'circularity',
        'aspect_ratio',
        'extent',
        'solidity'
    ]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # -------------------------------------------------------------------------
    # Edge Density
    # -------------------------------------------------------------------------
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)
    
    # -------------------------------------------------------------------------
    # Contour-based Features
    # -------------------------------------------------------------------------
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Use the largest contour (assumed to be the lesion)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Circularity: 4π × area / perimeter²
        # Perfect circle = 1.0, irregular shape < 1.0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Bounding box features
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # Convex hull solidity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Normalized values
        features.append(area / image.size)  # Normalized area
        features.append(perimeter / (image.shape[0] + image.shape[1]))  # Normalized perimeter
        features.append(circularity)
        features.append(aspect_ratio)
        features.append(extent)
        features.append(solidity)
    else:
        # No contours found - use default values
        features.extend([0, 0, 0, 1, 0, 0])
    
    return np.array(features), feature_names


# ============================================================================
# STATISTICAL FEATURES
# ============================================================================

def extract_statistical_features(image):
    """
    Extract statistical features from the image.
    
    These capture the overall distribution of pixel values and measure
    image complexity.
    
    FEATURES EXTRACTED:
    -------------------
    - Gray mean:    Average brightness
    - Gray std:     Brightness variation
    - Gray median:  Middle value (robust to outliers)
    - Gray range:   Max - min brightness
    - Skewness:     Asymmetry of brightness distribution
    - Kurtosis:     "Tailedness" of distribution
    - Entropy:      Randomness/complexity measure
    
    Total: 7 statistical features
    
    Parameters:
        image (numpy.ndarray): RGB image array
    
    Returns:
        features (numpy.ndarray): 1D array of statistical features
        feature_names (list): Names of each feature
    """
    features = []
    feature_names = [
        'gray_mean',
        'gray_std',
        'gray_median',
        'gray_range',
        'skewness',
        'kurtosis',
        'entropy'
    ]
    
    # Convert to grayscale and flatten
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).flatten().astype(float)
    
    # -------------------------------------------------------------------------
    # Basic Statistics
    # -------------------------------------------------------------------------
    features.append(np.mean(gray))
    features.append(np.std(gray))
    features.append(np.median(gray))
    features.append(np.ptp(gray))  # Peak-to-peak (range)
    
    # -------------------------------------------------------------------------
    # Higher-Order Moments
    # -------------------------------------------------------------------------
    mean = np.mean(gray)
    std = np.std(gray)
    
    if std > 0:
        # Skewness: measure of asymmetry
        # Positive = tail on right, Negative = tail on left
        skewness = np.mean(((gray - mean) / std) ** 3)
        
        # Kurtosis: measure of "tailedness"
        # High = heavy tails (outliers), Low = light tails
        kurtosis = np.mean(((gray - mean) / std) ** 4) - 3
    else:
        skewness = 0
        kurtosis = 0
    
    features.append(skewness)
    features.append(kurtosis)
    
    # -------------------------------------------------------------------------
    # Entropy (Information Content)
    # -------------------------------------------------------------------------
    # Higher entropy = more complex/random texture
    # Lower entropy = more uniform
    
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize to probability
    hist = hist[hist > 0]  # Remove zeros for log calculation
    entropy = -np.sum(hist * np.log2(hist))
    features.append(entropy)
    
    return np.array(features), feature_names


# ============================================================================
# COMBINED FEATURE EXTRACTION
# ============================================================================

def extract_all_features(image):
    """
    Extract ALL features from a single image.
    
    Combines:
    - Color features (36)
    - Texture features (10)
    - Shape features (7)
    - Statistical features (7)
    
    Total: 60 features per image
    
    Parameters:
        image (numpy.ndarray): RGB image array
    
    Returns:
        features (numpy.ndarray): 1D array of all features
        feature_names (list): Names of all features
    """
    # Resize image for consistent processing
    image = cv2.resize(image, IMAGE_SIZE)
    
    # Extract each feature type
    color_feats, color_names = extract_color_features(image)
    texture_feats, texture_names = extract_texture_features(image)
    shape_feats, shape_names = extract_shape_features(image)
    stat_feats, stat_names = extract_statistical_features(image)
    
    # Concatenate all features
    all_features = np.concatenate([color_feats, texture_feats, shape_feats, stat_feats])
    all_names = color_names + texture_names + shape_names + stat_names
    
    return all_features, all_names


def extract_features_from_dataset(images, labels):
    """
    Extract features from all images in a dataset.
    
    Parameters:
        images (list): List of image arrays
        labels (list): List of class labels
    
    Returns:
        X (numpy.ndarray): Feature matrix (num_samples × num_features)
        y (numpy.ndarray): Label array (num_samples,)
        feature_names (list): Names of all features
    """
    print("\n🔄 EXTRACTING FEATURES...")
    print("-" * 50)
    
    X = []
    feature_names = None
    total = len(images)
    
    for idx, image in enumerate(images):
        # Progress update
        if (idx + 1) % 100 == 0 or idx == 0 or idx == total - 1:
            print(f"   Processing: {idx + 1}/{total} ({100 * (idx + 1) / total:.1f}%)")
        
        features, names = extract_all_features(image)
        X.append(features)
        
        # Store feature names from first image
        if feature_names is None:
            feature_names = names
    
    X = np.array(X)
    y = np.array(labels)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Number of features: {len(feature_names)}")
    
    return X, y, feature_names


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_feature_extraction_example(image, save_path=None):
    """
    Visualize the feature extraction process on a single image.
    
    Creates a figure showing:
    - Original image
    - Color channels
    - Edge detection
    - Texture visualization
    
    Parameters:
        image (numpy.ndarray): RGB image array
        save_path (str, optional): Path to save the figure
    
    Returns:
        save_path (str): Path where figure was saved
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # RGB channels
    for i, (name, cmap) in enumerate([('Red', 'Reds'), ('Green', 'Greens'), ('Blue', 'Blues')]):
        axes[0, i + 1].imshow(image[:, :, i], cmap=cmap)
        axes[0, i + 1].set_title(f'{name} Channel', fontweight='bold')
        axes[0, i + 1].axis('off')
    
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    axes[1, 0].imshow(gray, cmap='gray')
    axes[1, 0].set_title('Grayscale', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    axes[1, 1].imshow(edges, cmap='gray')
    axes[1, 1].set_title('Edge Detection', fontweight='bold')
    axes[1, 1].axis('off')
    
    # HSV - Hue channel
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    axes[1, 2].imshow(hsv[:, :, 0], cmap='hsv')
    axes[1, 2].set_title('Hue (Color)', fontweight='bold')
    axes[1, 2].axis('off')
    
    # Histogram
    axes[1, 3].hist(gray.flatten(), bins=50, color='steelblue', edgecolor='white')
    axes[1, 3].set_title('Brightness Histogram', fontweight='bold')
    axes[1, 3].set_xlabel('Pixel Value')
    axes[1, 3].set_ylabel('Frequency')
    
    plt.suptitle('Feature Extraction Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "step2_feature_visualization.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved feature visualization to: {save_path}")
    
    return save_path


def print_feature_summary(feature_names):
    """
    Print a summary of all extracted features by category.
    """
    print("\n📋 FEATURE SUMMARY:")
    print("-" * 50)
    
    categories = {
        'Color': [n for n in feature_names if any(x in n for x in ['R_', 'G_', 'B_', 'H_', 'S_', 'V_', '_hist_'])],
        'Texture': [n for n in feature_names if 'glcm_' in n],
        'Shape': [n for n in feature_names if any(x in n for x in ['edge', 'area', 'perimeter', 'circular', 'aspect', 'extent', 'solidity'])],
        'Statistical': [n for n in feature_names if any(x in n for x in ['gray_', 'skew', 'kurt', 'entropy'])]
    }
    
    for category, feats in categories.items():
        print(f"\n   {category} Features ({len(feats)}):")
        for feat in feats[:5]:  # Show first 5
            print(f"      • {feat}")
        if len(feats) > 5:
            print(f"      ... and {len(feats) - 5} more")
    
    print(f"\n   Total Features: {len(feature_names)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Demonstrate the feature extraction functionality.
    """
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE EXTRACTION")
    print("=" * 70)
    print("""
    This step extracts meaningful numerical features from images:
    
    🎨 COLOR FEATURES:
       RGB and HSV statistics, color histograms
       
    🔳 TEXTURE FEATURES:
       GLCM-based descriptors (contrast, homogeneity, etc.)
       
    📐 SHAPE FEATURES:
       Edge density, circularity, contour properties
       
    📊 STATISTICAL FEATURES:
       Entropy, skewness, kurtosis
    """)
    
    # Load sample data
    from step1_load_data import load_images_with_labels
    from config import TRAIN_PATH, MAX_TRAIN_SAMPLES
    
    images, labels, paths = load_images_with_labels(TRAIN_PATH, MAX_TRAIN_SAMPLES)
    
    # Visualize feature extraction on one image
    if images:
        visualize_feature_extraction_example(images[0])
    
    # Extract features from dataset
    X, y, feature_names = extract_features_from_dataset(images, labels)
    
    # Print feature summary
    print_feature_summary(feature_names)
    
    print("\n✅ STEP 2 COMPLETE!")
    print(f"   - Feature matrix shape: {X.shape}")
    print(f"   - Number of features per image: {len(feature_names)}")
    
    return X, y, feature_names


if __name__ == "__main__":
    main()
