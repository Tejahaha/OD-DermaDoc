"""
=============================================================================
STEP 1: LOADING IMAGES AND LABELS
=============================================================================

This module handles loading images from the dataset along with their 
corresponding class labels.

Key Functions:
    - load_images_with_labels(): Load images and parse label files
    - get_class_distribution(): Analyze class distribution in dataset
    - visualize_sample_images(): Display sample images from each class

Usage:
    from step1_load_data import load_images_with_labels, visualize_sample_images
    
    images, labels, paths = load_images_with_labels(TRAIN_PATH)
    visualize_sample_images(images, labels)
"""

import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from collections import Counter

# Import configuration
from config import (
    TRAIN_PATH, VALID_PATH, TEST_PATH, OUTPUT_DIR,
    IMAGE_EXTENSIONS, RANDOM_SEED
)

# Set random seed
np.random.seed(RANDOM_SEED)


# ============================================================================
# MAIN DATA LOADING FUNCTION
# ============================================================================

def load_images_with_labels(split_path, max_samples=None):
    """
    Load images and their corresponding class labels from a dataset split.
    
    DATASET STRUCTURE EXPECTED:
        split/
            images/
                image1.jpg
                image2.jpg
                ...
            labels/
                image1.txt  (contains: class_id x1 y1 x2 y2 ...)
                image2.txt
                ...
    
    The class_id is extracted from the first number in each label file.
    
    Parameters:
        split_path (str): Path to the split folder (train/valid/test)
        max_samples (int, optional): Maximum number of samples to load.
                                     If None, loads all samples.
    
    Returns:
        images (list): List of loaded images as numpy arrays (RGB format)
        labels (list): List of corresponding class IDs (integers)
        paths (list): List of file paths for each image
    
    Example:
        >>> images, labels, paths = load_images_with_labels("train/", max_samples=100)
        >>> print(f"Loaded {len(images)} images")
        Loaded 100 images
    """
    images_dir = os.path.join(split_path, "images")
    labels_dir = os.path.join(split_path, "labels")
    
    # Find all image files
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    # Limit samples if specified
    if max_samples is not None and len(image_files) > max_samples:
        image_files = image_files[:max_samples]
    
    images = []
    labels = []
    paths = []
    
    print(f"📂 Loading images from: {split_path}")
    print(f"   Found {len(image_files)} image files")
    
    # Progress tracking
    total = len(image_files)
    
    for idx, img_path in enumerate(image_files):
        # Progress update every 100 images
        if (idx + 1) % 100 == 0:
            print(f"   Loading: {idx + 1}/{total} ({100 * (idx + 1) / total:.1f}%)")
        
        # Get corresponding label file
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        # Skip if label file doesn't exist
        if not os.path.exists(label_path):
            continue
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Parse label file to extract class ID
        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                parts = first_line.split()
                class_id = int(parts[0])
            else:
                continue
        
        images.append(image)
        labels.append(class_id)
        paths.append(img_path)
    
    print(f"   ✅ Successfully loaded {len(images)} images with labels")
    
    return images, labels, paths


# ============================================================================
# DATA ANALYSIS FUNCTIONS
# ============================================================================

def get_class_distribution(labels):
    """
    Analyze the distribution of classes in the dataset.
    
    Parameters:
        labels (list): List of class IDs
    
    Returns:
        class_counts (Counter): Dictionary-like object with class counts
        unique_classes (list): Sorted list of unique class IDs
    
    Example:
        >>> class_counts, unique = get_class_distribution([0, 1, 0, 2, 1, 1])
        >>> print(class_counts)
        Counter({1: 3, 0: 2, 2: 1})
    """
    class_counts = Counter(labels)
    unique_classes = sorted(class_counts.keys())
    
    return class_counts, unique_classes


def print_class_distribution(labels, split_name="Dataset"):
    """
    Print a formatted class distribution report.
    
    Parameters:
        labels (list): List of class IDs
        split_name (str): Name of the dataset split for display
    """
    class_counts, unique_classes = get_class_distribution(labels)
    
    print(f"\n📊 CLASS DISTRIBUTION ({split_name}):")
    print("-" * 50)
    
    max_count = max(class_counts.values())
    bar_scale = 30 / max_count  # Scale bars to max 30 characters
    
    for cls in unique_classes:
        count = class_counts[cls]
        bar = "█" * int(count * bar_scale)
        print(f"   Class {cls:2d}: {count:5d} samples {bar}")
    
    print("-" * 50)
    print(f"   Total:   {sum(class_counts.values()):5d} samples")
    print(f"   Classes: {len(unique_classes)}")
    
    return class_counts, unique_classes


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_sample_images(images, labels, num_samples=12, save_path=None):
    """
    Create a grid visualization of sample images with their class labels.
    
    Parameters:
        images (list): List of images (numpy arrays)
        labels (list): List of corresponding class labels
        num_samples (int): Number of samples to display
        save_path (str, optional): Path to save the figure. 
                                   If None, saves to default location.
    
    Returns:
        save_path (str): Path where the figure was saved
    """
    print("\n🖼️  Visualizing sample images...")
    
    # Set up the grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    # Select random indices
    num_to_show = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_to_show, replace=False)
    
    # Color map for different classes
    unique_classes = sorted(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
    class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    
    # Display images
    for ax_idx, sample_idx in enumerate(indices):
        image = images[sample_idx]
        label = labels[sample_idx]
        
        axes[ax_idx].imshow(image)
        axes[ax_idx].set_title(
            f"Class {label}", 
            fontsize=12, 
            fontweight='bold',
            color=class_to_color[label][:3]  # RGB without alpha
        )
        axes[ax_idx].axis('off')
        
        # Add colored border
        for spine in axes[ax_idx].spines.values():
            spine.set_edgecolor(class_to_color[label])
            spine.set_linewidth(3)
    
    # Hide unused subplots
    for idx in range(num_to_show, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Sample Images with Class Labels", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "step1_sample_images.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved visualization to: {save_path}")
    
    return save_path


def visualize_class_distribution(labels, save_path=None):
    """
    Create a bar chart visualization of class distribution.
    
    Parameters:
        labels (list): List of class labels
        save_path (str, optional): Path to save the figure
    
    Returns:
        save_path (str): Path where the figure was saved
    """
    class_counts, unique_classes = get_class_distribution(labels)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar chart
    counts = [class_counts[cls] for cls in unique_classes]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(unique_classes)))
    
    bars = ax.bar([f"Class {c}" for c in unique_classes], counts, color=colors)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "step1_class_distribution.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved class distribution chart to: {save_path}")
    
    return save_path


# ============================================================================
# MAIN EXECUTION (when run directly)
# ============================================================================

def main():
    """
    Demonstrate the data loading functionality.
    """
    print("\n" + "=" * 70)
    print("STEP 1: LOADING IMAGES AND LABELS")
    print("=" * 70)
    print("""
    This step loads images from the dataset and extracts class labels
    from the corresponding label files.
    
    What we're doing:
    1. Find all image files in the images/ folder
    2. Parse label files to get class IDs
    3. Load images and convert to RGB format
    4. Analyze class distribution
    5. Visualize sample images
    """)
    
    # Load training data
    from config import MAX_TRAIN_SAMPLES
    images, labels, paths = load_images_with_labels(TRAIN_PATH, MAX_TRAIN_SAMPLES)
    
    # Print class distribution
    class_counts, unique_classes = print_class_distribution(labels, "Training Set")
    
    # Visualize samples
    visualize_sample_images(images, labels)
    visualize_class_distribution(labels)
    
    print("\n✅ STEP 1 COMPLETE!")
    print(f"   - Loaded {len(images)} images")
    print(f"   - Found {len(unique_classes)} unique classes: {unique_classes}")
    print(f"   - Visualizations saved to: {OUTPUT_DIR}")
    
    return images, labels, paths


if __name__ == "__main__":
    main()
