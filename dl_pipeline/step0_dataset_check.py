"""
=============================================================================
STEP 0: DATASET CHECK & STATISTICS
=============================================================================

Verify your dataset structure and visualize sample segmentations.
This helps ensure your data is correctly formatted for YOLOv8.

Usage:
    python step0_dataset_check.py
"""

import os
import sys
import glob
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from dl_pipeline.config import (
    TRAIN_PATH, VALID_PATH, TEST_PATH, OUTPUT_DIR,
    CLASS_NAMES, CLASS_COLORS, NUM_CLASSES, IMAGE_EXTENSIONS,
    RANDOM_SEED
)

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_image_label_pairs(split_path):
    """
    Find all image-label pairs in a split folder.
    
    YOLO expects this structure:
        split/
            images/
                image1.jpg
            labels/
                image1.txt
    
    Returns: List of (image_path, label_path) tuples
    """
    images_dir = os.path.join(split_path, "images")
    labels_dir = os.path.join(split_path, "labels")
    
    pairs = []
    
    # Find all images
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    
    for img_path in image_files:
        # Get corresponding label file
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))
    
    return pairs


def parse_yolo_segmentation_label(label_path, img_width, img_height):
    """
    Parse a YOLO segmentation label file.
    
    YOLO format: class_id x1 y1 x2 y2 x3 y3 ... (normalized 0-1)
    
    Returns: List of (class_id, polygon_points)
    """
    objects = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # Need at least class_id + 3 points
                continue
            
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Group into (x, y) pairs
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = coords[i] * img_width
                    y = coords[i + 1] * img_height
                    points.append([x, y])
            
            if len(points) >= 3:
                objects.append((class_id, np.array(points, dtype=np.int32)))
    
    return objects


def visualize_segmentation(image, objects):
    """Overlay segmentation polygons on an image."""
    vis_image = image.copy()
    overlay = vis_image.copy()
    
    for class_id, polygon in objects:
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        
        # Draw filled polygon
        cv2.fillPoly(overlay, [polygon], color)
        
        # Draw outline
        cv2.polylines(vis_image, [polygon], True, color, 2)
        
        # Add label
        centroid = polygon.mean(axis=0).astype(int)
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        cv2.putText(vis_image, class_name, tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Blend overlay
    vis_image = cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0)
    
    return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def check_dataset_structure():
    """Verify YOLO folder structure exists."""
    print("\n" + "=" * 60)
    print("CHECKING DATASET STRUCTURE")
    print("=" * 60)
    
    splits = {"Train": TRAIN_PATH, "Valid": VALID_PATH, "Test": TEST_PATH}
    all_good = True
    
    for name, path in splits.items():
        images_dir = os.path.join(path, "images")
        labels_dir = os.path.join(path, "labels")
        
        images_ok = os.path.exists(images_dir)
        labels_ok = os.path.exists(labels_dir)
        
        status = "✅" if (images_ok and labels_ok) else "❌"
        print(f"  {status} {name}:")
        print(f"      images/: {'Found' if images_ok else 'MISSING'}")
        print(f"      labels/: {'Found' if labels_ok else 'MISSING'}")
        
        if not (images_ok and labels_ok):
            all_good = False
    
    return all_good


def get_dataset_statistics():
    """Get comprehensive dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    splits = {"Train": TRAIN_PATH, "Valid": VALID_PATH, "Test": TEST_PATH}
    all_stats = {}
    
    for name, path in splits.items():
        pairs = get_image_label_pairs(path)
        
        class_counts = {i: 0 for i in range(NUM_CLASSES)}
        objects_per_image = []
        
        for img_path, label_path in pairs[:100]:  # Sample for speed
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            h, w = image.shape[:2]
            objects = parse_yolo_segmentation_label(label_path, w, h)
            objects_per_image.append(len(objects))
            
            for class_id, _ in objects:
                if class_id < NUM_CLASSES:
                    class_counts[class_id] += 1
        
        all_stats[name] = {
            "total": len(pairs),
            "class_counts": class_counts,
            "avg_objects": np.mean(objects_per_image) if objects_per_image else 0
        }
        
        print(f"\n  📁 {name}: {len(pairs)} images")
        print(f"      Avg objects per image: {all_stats[name]['avg_objects']:.2f}")
    
    # Print class distribution
    print("\n  📊 Class Distribution (from sample):")
    train_counts = all_stats["Train"]["class_counts"]
    for class_id, count in train_counts.items():
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        bar = "█" * min(count // 2, 30)
        print(f"      {class_id}: {class_name:20} {count:4} {bar}")
    
    return all_stats


def visualize_samples(num_samples=6):
    """Visualize random samples with segmentation overlays."""
    print("\n" + "=" * 60)
    print("VISUALIZING SAMPLES")
    print("=" * 60)
    
    pairs = get_image_label_pairs(TRAIN_PATH)
    if not pairs:
        print("  ❌ No image-label pairs found!")
        return
    
    selected = random.sample(pairs, min(num_samples, len(pairs)))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (img_path, label_path) in enumerate(selected):
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        
        objects = parse_yolo_segmentation_label(label_path, w, h)
        vis_image = visualize_segmentation(image, objects)
        
        axes[idx].imshow(vis_image)
        axes[idx].set_title(f"{os.path.basename(img_path)}\n{len(objects)} object(s)")
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(selected), 6):
        axes[idx].axis('off')
    
    plt.suptitle("Sample Segmentation Visualizations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "step0_samples.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  ✅ Saved visualization to: {output_path}")


def run_dataset_check():
    """Run complete dataset check."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " STEP 0: DATASET CHECK & STATISTICS ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Check structure
    structure_ok = check_dataset_structure()
    if not structure_ok:
        print("\n  ❌ Dataset structure issues found. Please fix before training.")
        return False
    
    # Get statistics
    stats = get_dataset_statistics()
    
    # Visualize samples
    visualize_samples()
    
    print("\n" + "=" * 60)
    print("✅ STEP 0 COMPLETE!")
    print("   Dataset is ready for training.")
    print("=" * 60 + "\n")
    
    return True


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_dataset_check()
