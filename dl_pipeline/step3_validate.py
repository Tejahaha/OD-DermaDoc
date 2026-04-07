"""
=============================================================================
STEP 3: VALIDATE & VISUALIZE MODEL PREDICTIONS
=============================================================================

Load the trained YOLOv8 model and assess its quality on held-out images.

What this module does
---------------------
``run_validation``
    1. Loads the best model weights produced by Step 2.
    2. Randomly selects images from the validation (or test) split.
    3. Runs inference on each image and renders the predicted segmentation
       masks as semi-transparent overlays with class labels and confidence
       scores.
    4. Saves a 2×3 image grid to ``results/step3_validation.png``.
    5. Calls ``model.val()`` to compute dataset-wide metrics
       (mAP50 and mAP50-95) and prints them to stdout.

Visualisation details
---------------------
Each predicted mask is:
* Filled with the class colour at 40 % opacity so the skin texture remains
  visible beneath the overlay.
* Outlined with a 2-pixel solid contour.
* Labelled with ``<ClassName>: <confidence>`` at the top of the lesion area.
Predictions below ``config.CONF_THRESHOLD`` are suppressed.

Command-line usage
------------------
    # Use the default best.pt from Step 2
    python step3_validate.py

    # Specify a custom model path
    python step3_validate.py --model path/to/weights.pt

    # Visualise more images
    python step3_validate.py --num-samples 9

    # Use the test split instead of the validation split
    python step3_validate.py --test
"""

import os
import sys
import argparse
import glob
import random
from typing import List, Optional

# Make the parent package importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend; safe for headless servers.
import matplotlib.pyplot as plt

from dl_pipeline.config import (
    VALID_PATH, TEST_PATH, OUTPUT_DIR, IMAGE_EXTENSIONS,
    CLASS_NAMES, CLASS_COLORS, CONF_THRESHOLD, RANDOM_SEED
)

# Fix the random seed so the same validation images are selected on every run.
random.seed(RANDOM_SEED)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_default_model_path() -> str:
    """
    Return the expected path to the best model weights from Step 2.

    Returns
    -------
    str
        Absolute path to ``<OUTPUT_DIR>/yolov8_seg/weights/best.pt``.
    """
    return os.path.join(OUTPUT_DIR, "yolov8_seg", "weights", "best.pt")


def load_model(model_path: Optional[str] = None):
    """
    Load a trained YOLOv8 model from disk.

    Uses the default ``best.pt`` produced by Step 2 when no explicit path
    is provided.  Prints a clear error and returns ``None`` if the file is
    missing so callers can handle the failure gracefully.

    Parameters
    ----------
    model_path : str, optional
        Path to the ``.pt`` weights file.  Defaults to
        ``<OUTPUT_DIR>/yolov8_seg/weights/best.pt``.

    Returns
    -------
    ultralytics.YOLO or None
        Loaded model object, or ``None`` if the weights file was not found.
    """
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


def get_validation_images(
    split_path: str,
    num_samples: int = 6,
) -> List[str]:
    """
    Randomly sample image paths from a dataset split.

    Scans ``<split_path>/images/`` for all supported image extensions and
    returns a random subset of up to ``num_samples`` paths.

    Parameters
    ----------
    split_path : str
        Absolute path to the split directory (e.g. ``VALID_PATH``).
    num_samples : int, optional
        Maximum number of images to return.  Actual count is capped at
        the number of images available.  Default is ``6``.

    Returns
    -------
    list of str
        Randomly selected image file paths.  Empty list if the directory
        contains no supported images.
    """
    images_dir   = os.path.join(split_path, "images")
    image_files: List[str] = []

    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))

    if not image_files:
        return []

    return random.sample(image_files, min(num_samples, len(image_files)))


def visualize_prediction(
    image: np.ndarray,
    results,
    show_conf: bool = True,
) -> np.ndarray:
    """
    Overlay model predictions on an image as coloured segmentation masks.

    For each detected lesion the function:
    * Resizes the predicted binary mask to match the original image resolution.
    * Applies a filled, semi-transparent colour patch on a blend layer.
    * Draws a solid 2-pixel contour around the lesion boundary.
    * Optionally adds a ``<ClassName>: <conf>`` text label near the top of
      the lesion (controlled by ``show_conf``).
    Detections below ``CONF_THRESHOLD`` are skipped.

    Parameters
    ----------
    image : numpy.ndarray
        Original BGR image loaded with ``cv2.imread``.
    results : ultralytics.engine.results.Results
        Inference result for a single image, as returned by ``model(img)[0]``.
    show_conf : bool, optional
        When ``True`` (default), draw the class name and confidence score
        on each detected lesion.

    Returns
    -------
    numpy.ndarray
        Annotated image in **RGB** format (suitable for Matplotlib).
    """
    vis_image = image.copy()

    # If the model found no segmentation masks, return the plain image.
    if results.masks is None:
        return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

    masks  = results.masks.data.cpu().numpy()   # Shape: (N, H_mask, W_mask)
    boxes  = results.boxes.data.cpu().numpy()   # Shape: (N, 6)  [x1,y1,x2,y2,conf,cls]
    overlay = vis_image.copy()                  # Separate layer for alpha-blended fills.

    for mask, box in zip(masks, boxes):
        conf     = box[4]
        class_id = int(box[5])

        # Skip low-confidence detections.
        if conf < CONF_THRESHOLD:
            continue

        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

        # Upscale the mask to the original image dimensions.
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask_bool    = mask_resized > 0.5   # Binarise at 50 % threshold.

        # Flood the mask area on the overlay layer with the class colour.
        overlay[mask_bool] = color

        # Draw a solid outline using the extracted mask contours.
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_image, contours, -1, color, 2)

        # Add a text label anchored to the topmost pixel of the mask.
        if show_conf:
            class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
            label      = f"{class_name}: {conf:.2f}"

            y_coords, x_coords = np.where(mask_bool)
            if len(y_coords) > 0:
                # Position the label above (or at) the topmost mask pixel.
                text_y = max(20, min(y_coords))
                text_x = min(x_coords)
                cv2.putText(vis_image, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Alpha-blend the filled overlay (40 %) with the outlined image (60 %).
    vis_image = cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0)

    # Convert BGR → RGB for Matplotlib display.
    return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)


# ============================================================================
# VALIDATION ORCHESTRATOR
# ============================================================================

def run_validation(
    model_path: Optional[str] = None,
    num_samples: int = 6,
    use_test: bool = False,
):
    """
    Validate the trained model: visualise predictions and compute metrics.

    Steps performed:
    1. Load the model (defaults to ``best.pt`` from Step 2).
    2. Sample images from the validation or test split.
    3. Run inference on each image and build a 2×3 visualisation grid.
    4. Save the grid to ``results/step3_validation.png``.
    5. Call ``model.val()`` for dataset-wide mAP50 and mAP50-95 metrics.

    Parameters
    ----------
    model_path : str, optional
        Path to model weights.  Defaults to
        ``<OUTPUT_DIR>/yolov8_seg/weights/best.pt``.
    num_samples : int, optional
        Number of images to include in the visualisation grid.
        Default is ``6`` (fills a 2×3 grid perfectly).
    use_test : bool, optional
        When ``True``, sample from ``TEST_PATH`` instead of ``VALID_PATH``.
        Default is ``False``.

    Returns
    -------
    ultralytics.utils.metrics.SegmentMetrics or None
        Metrics object from ``model.val()``, or ``None`` if the model
        could not be loaded.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " STEP 3: VALIDATE & VISUALIZE PREDICTIONS ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    # Load the trained model; bail out cleanly if it's missing.
    model = load_model(model_path)
    if model is None:
        return

    # Choose the dataset split for visualisation.
    split_path = TEST_PATH  if use_test else VALID_PATH
    split_name = "Test"     if use_test else "Validation"

    print(f"\n  📁 Using {split_name} set: {split_path}")

    image_paths = get_validation_images(split_path, num_samples)
    if not image_paths:
        print(f"  ❌ No images found in {split_path}")
        return

    print(f"  📸 Running inference on {len(image_paths)} images...")

    # Build the visualisation grid (2 rows × 3 columns).
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, img_path in enumerate(image_paths):
        if idx >= 6:
            break   # Grid is full; ignore any extra image paths.

        image   = cv2.imread(img_path)
        results = model(img_path, verbose=False)[0]  # Run inference silently.

        vis_image      = visualize_prediction(image, results)
        num_detections = len(results.boxes) if results.boxes is not None else 0

        axes[idx].imshow(vis_image)
        axes[idx].set_title(f"{os.path.basename(img_path)}\n{num_detections} detection(s)")
        axes[idx].axis('off')

    # Hide any unused subplot panels.
    for idx in range(len(image_paths), 6):
        axes[idx].axis('off')

    plt.suptitle(f"{split_name} Set Predictions", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "step3_validation.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\n  ✅ Saved visualization to: {output_path}")

    # Compute full dataset-level segmentation metrics.
    print("\n  📊 Computing validation metrics...")
    metrics = model.val(verbose=False)

    print(f"\n  📊 Validation Metrics:")
    print(f"     mAP50:    {metrics.seg.map50:.4f}")
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
        use_test=args.test,
    )
