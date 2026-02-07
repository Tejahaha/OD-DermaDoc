"""
=============================================================================
STEP 3: VALIDATE & VISUALIZE MODEL PREDICTIONS
=============================================================================

Load the trained model and visualize predictions on validation images.
This helps assess model quality and identify failure cases.

Usage:
    python step3_validate.py
    
    # Or with custom model path:
    python step3_validate.py --model path/to/weights.pt
"""

import os
import sys
import argparse
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dl_pipeline.config import (
    VALID_PATH, TEST_PATH, OUTPUT_DIR, IMAGE_EXTENSIONS,
    CLASS_NAMES, CLASS_COLORS, CONF_THRESHOLD, RANDOM_SEED
)

# Set random seed
random.seed(RANDOM_SEED)


def get_default_model_path():
    """Get the default trained model path."""
    return os.path.join(OUTPUT_DIR, "yolov8_seg", "weights", "best.pt")


def load_model(model_path=None):
    """Load the trained YOLOv8 model."""
    from ultralytics import YOLO
    
    if model_path is None:
        model_path = get_default_model_path()
    
    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        print("     Run step2_train.py first!")
        return None
    
    print(f"  ⏳ Loading model: {model_path}")
    model = YOLO(model_path)
    print("  ✅ Model loaded!")
    
    return model


def get_validation_images(split_path, num_samples=6):
    """Get random validation images."""
    import glob
    
    images_dir = os.path.join(split_path, "images")
    image_files = []
    
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    
    if not image_files:
        return []
    
    return random.sample(image_files, min(num_samples, len(image_files)))


def visualize_prediction(image, results, show_conf=True):
    """Overlay model predictions on an image."""
    vis_image = image.copy()
    
    if results.masks is None:
        return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    
    masks = results.masks.data.cpu().numpy()
    boxes = results.boxes.data.cpu().numpy()
    
    overlay = vis_image.copy()
    
    for mask, box in zip(masks, boxes):
        conf = box[4]
        class_id = int(box[5])
        
        if conf < CONF_THRESHOLD:
            continue
        
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        
        # Resize mask to image size
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask_bool = mask_resized > 0.5
        
        # Draw filled mask
        overlay[mask_bool] = color
        
        # Draw contour
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_image, contours, -1, color, 2)
        
        # Add label
        if show_conf:
            class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
            label = f"{class_name}: {conf:.2f}"
            
            y_coords, x_coords = np.where(mask_bool)
            if len(y_coords) > 0:
                text_y = max(20, min(y_coords))
                text_x = min(x_coords)
                cv2.putText(vis_image, label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Blend overlay
    vis_image = cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0)
    
    return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)


def run_validation(model_path=None, num_samples=6, use_test=False):
    """
    Run validation and create visualizations.
    
    Args:
        model_path: Path to model weights (None uses default)
        num_samples: Number of images to visualize
        use_test: Use test set instead of validation set
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " STEP 3: VALIDATE & VISUALIZE PREDICTIONS ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Get images
    split_path = TEST_PATH if use_test else VALID_PATH
    split_name = "Test" if use_test else "Validation"
    
    print(f"\n  📁 Using {split_name} set: {split_path}")
    
    image_paths = get_validation_images(split_path, num_samples)
    if not image_paths:
        print(f"  ❌ No images found in {split_path}")
        return
    
    print(f"  📸 Running inference on {len(image_paths)} images...")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, img_path in enumerate(image_paths):
        if idx >= 6:
            break
        
        # Load image
        image = cv2.imread(img_path)
        
        # Run inference
        results = model(img_path, verbose=False)[0]
        
        # Visualize
        vis_image = visualize_prediction(image, results)
        
        # Count detections
        num_detections = len(results.boxes) if results.boxes is not None else 0
        
        axes[idx].imshow(vis_image)
        axes[idx].set_title(f"{os.path.basename(img_path)}\n{num_detections} detection(s)")
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(image_paths), 6):
        axes[idx].axis('off')
    
    plt.suptitle(f"{split_name} Set Predictions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "step3_validation.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\n  ✅ Saved visualization to: {output_path}")
    
    # Run full validation for metrics
    print("\n  📊 Computing validation metrics...")
    metrics = model.val(verbose=False)
    
    print(f"\n  📊 Validation Metrics:")
    print(f"     mAP50: {metrics.seg.map50:.4f}")
    print(f"     mAP50-95: {metrics.seg.map:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ STEP 3 COMPLETE!")
    print(f"   Visualizations saved to: {output_path}")
    print("=" * 60 + "\n")
    
    return metrics


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLOv8 Model")
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights')
    parser.add_argument('--num-samples', type=int, default=6,
                        help='Number of images to visualize')
    parser.add_argument('--test', action='store_true',
                        help='Use test set instead of validation')
    args = parser.parse_args()
    
    run_validation(
        model_path=args.model,
        num_samples=args.num_samples,
        use_test=args.test
    )
