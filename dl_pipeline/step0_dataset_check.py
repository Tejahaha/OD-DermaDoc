"""
=============================================================================
STEP 0: DATASET CHECK & STATISTICS
=============================================================================

Verify that the ISIC skin-lesion dataset is correctly laid out for YOLOv8
and produce diagnostic plots before any expensive training begins.

What this module does
---------------------
1. ``check_dataset_structure`` – confirm that every split
   (train / valid / test) has both an ``images/`` and a ``labels/``
   sub-directory.
2. ``get_dataset_statistics`` – count images per split, measure the
   class distribution (sampled for speed), and report average objects
   per image.
3. ``visualize_samples`` – load six random training images, render their
   ground-truth segmentation polygons as coloured overlays, and save the
   figure to ``results/step0_samples.png``.

Expected dataset layout (YOLO segmentation format)
---------------------------------------------------
    <DATASET_PATH>/
        train/
            images/   (*.jpg, *.png, …)
            labels/   (*.txt  – one file per image)
        valid/
            images/
            labels/
        test/
            images/
            labels/

Label file format (one line per object)
----------------------------------------
    <class_id> <x1> <y1> <x2> <y2> … <xN> <yN>

All coordinates are normalised to [0, 1] relative to the image dimensions.

Command-line usage
------------------
    python step0_dataset_check.py
"""

import os
import sys
import glob
import random
from typing import Dict, List, Optional, Tuple

# Make the parent package importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend; avoids display errors on headless servers.
import matplotlib.pyplot as plt

from dl_pipeline.config import (
    TRAIN_PATH, VALID_PATH, TEST_PATH, OUTPUT_DIR,
    CLASS_NAMES, CLASS_COLORS, NUM_CLASSES, IMAGE_EXTENSIONS,
    RANDOM_SEED
)

# Seed both random libraries so sample selection is deterministic.
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_image_label_pairs(split_path: str) -> List[Tuple[str, str]]:
    """
    Collect all matched image–label file pairs inside a dataset split.

    Scans ``<split_path>/images/`` for every supported image extension and
    looks for a corresponding ``.txt`` annotation file in
    ``<split_path>/labels/``.  Images without a matching label are silently
    skipped, which keeps the pipeline tolerant of partially annotated splits.

    YOLO expected structure
    -----------------------
    ::

        split/
            images/   image1.jpg  image2.png  …
            labels/   image1.txt  image2.txt  …

    Parameters
    ----------
    split_path : str
        Absolute path to a dataset split directory (e.g. ``TRAIN_PATH``).

    Returns
    -------
    list of (str, str)
        Each tuple is ``(image_path, label_path)`` for a matched pair.
        Only pairs where both files exist are returned.
    """
    images_dir = os.path.join(split_path, "images")
    labels_dir = os.path.join(split_path, "labels")

    pairs: List[Tuple[str, str]] = []

    # Accumulate image paths for every supported extension.
    image_files: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))

    for img_path in image_files:
        # Strip the file extension to get the shared base name (e.g. "ISIC_0024348").
        img_name   = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{img_name}.txt")

        if os.path.exists(label_path):
            pairs.append((img_path, label_path))

    return pairs


def parse_yolo_segmentation_label(
    label_path: str,
    img_width: int,
    img_height: int,
) -> List[Tuple[int, np.ndarray]]:
    """
    Parse a YOLO segmentation annotation file into pixel-space polygons.

    Each line in the file encodes one annotated object:
    ::

        <class_id>  <x1> <y1>  <x2> <y2>  …  <xN> <yN>

    All coordinates are normalised floats in [0, 1].  This function
    de-normalises them to absolute pixel coordinates using the supplied
    image dimensions.

    Lines with fewer than 7 tokens (class_id + 3 points minimum) are
    silently skipped to handle malformed annotations gracefully.

    Parameters
    ----------
    label_path : str
        Path to the YOLO ``.txt`` annotation file.
    img_width : int
        Width of the corresponding image in pixels.
    img_height : int
        Height of the corresponding image in pixels.

    Returns
    -------
    list of (int, numpy.ndarray)
        Each element is ``(class_id, polygon)``, where ``polygon`` is an
        ``(N, 2)`` integer array of ``[x, y]`` pixel coordinates suitable
        for ``cv2.fillPoly`` / ``cv2.polylines``.
    """
    objects: List[Tuple[int, np.ndarray]] = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Need at least class_id + 3 coordinate pairs (7 tokens total).
            if len(parts) < 7:
                continue

            class_id = int(parts[0])
            # Remaining tokens are alternating normalised x, y coordinates.
            coords = list(map(float, parts[1:]))

            # Re-group flat coordinate list into (x, y) pixel pairs.
            points: List[List[float]] = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = coords[i]     * img_width
                    y = coords[i + 1] * img_height
                    points.append([x, y])

            # A valid polygon requires at least 3 vertices.
            if len(points) >= 3:
                objects.append((class_id, np.array(points, dtype=np.int32)))

    return objects


def visualize_segmentation(
    image: np.ndarray,
    objects: List[Tuple[int, np.ndarray]],
) -> np.ndarray:
    """
    Render ground-truth segmentation polygons as semi-transparent overlays.

    For each annotated object the function:
    * Draws a filled, semi-transparent coloured polygon on a blend layer.
    * Draws a solid polygon outline on the main image.
    * Places a text label at the polygon's centroid.

    The blend weight (0.4 overlay, 0.6 original) keeps the underlying
    skin texture visible while clearly delineating lesion boundaries.

    Parameters
    ----------
    image : numpy.ndarray
        Original BGR image loaded with ``cv2.imread``.
    objects : list of (int, numpy.ndarray)
        Ground-truth objects as returned by
        :func:`parse_yolo_segmentation_label`.

    Returns
    -------
    numpy.ndarray
        Annotated image converted to **RGB** for display with Matplotlib.
    """
    vis_image = image.copy()
    overlay   = vis_image.copy()   # Separate layer for semi-transparent fills.

    for class_id, polygon in objects:
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

        # Fill the polygon on the overlay layer only.
        cv2.fillPoly(overlay, [polygon], color)

        # Draw a solid 2-pixel outline on the main image.
        cv2.polylines(vis_image, [polygon], True, color, 2)

        # Place the class label at the polygon's geometric centroid.
        centroid   = polygon.mean(axis=0).astype(int)
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        cv2.putText(vis_image, class_name, tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Blend: 40 % filled colour + 60 % original details.
    vis_image = cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0)

    # Convert BGR → RGB so Matplotlib renders colours correctly.
    return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def check_dataset_structure() -> bool:
    """
    Verify that every dataset split has the required sub-directories.

    Checks for ``images/`` and ``labels/`` inside each of the three splits
    (train, valid, test) and prints a ✅ / ❌ status for each.

    Returns
    -------
    bool
        ``True`` only if *all* six directories (two per split) exist.
        ``False`` as soon as any required directory is missing.
    """
    print("\n" + "=" * 60)
    print("CHECKING DATASET STRUCTURE")
    print("=" * 60)

    splits     = {"Train": TRAIN_PATH, "Valid": VALID_PATH, "Test": TEST_PATH}
    all_good   = True

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


def get_dataset_statistics() -> Dict[str, Dict]:
    """
    Compute and print summary statistics for each dataset split.

    For each split the function:
    * Counts the total number of matched image–label pairs.
    * Samples up to 100 images to tally per-class object counts and compute
      the average number of annotated objects per image (sampling avoids
      long run-times on large datasets).
    * Prints a horizontal bar chart of the training-set class distribution.

    Returns
    -------
    dict
        Keyed by split name (``"Train"``, ``"Valid"``, ``"Test"``).
        Each value is a dict with:
        * ``"total"`` (int)        – total image–label pairs in the split.
        * ``"class_counts"`` (dict) – ``{class_id: object_count}`` from the sample.
        * ``"avg_objects"`` (float) – mean objects per sampled image.
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    splits    = {"Train": TRAIN_PATH, "Valid": VALID_PATH, "Test": TEST_PATH}
    all_stats: Dict[str, Dict] = {}

    for name, path in splits.items():
        pairs = get_image_label_pairs(path)

        # Initialise per-class counters to zero for all known classes.
        class_counts        = {i: 0 for i in range(NUM_CLASSES)}
        objects_per_image   = []

        # Sample at most 100 images to keep this step fast.
        for img_path, label_path in pairs[:100]:
            image = cv2.imread(img_path)
            if image is None:
                continue    # Skip unreadable images silently.

            h, w = image.shape[:2]
            objects = parse_yolo_segmentation_label(label_path, w, h)
            objects_per_image.append(len(objects))

            for class_id, _ in objects:
                if class_id < NUM_CLASSES:
                    class_counts[class_id] += 1

        all_stats[name] = {
            "total":        len(pairs),
            "class_counts": class_counts,
            "avg_objects":  np.mean(objects_per_image) if objects_per_image else 0,
        }

        print(f"\n  📁 {name}: {len(pairs)} images")
        print(f"      Avg objects per image: {all_stats[name]['avg_objects']:.2f}")

    # Print an ASCII bar chart for the training split's class distribution.
    print("\n  📊 Class Distribution (from sample):")
    train_counts = all_stats["Train"]["class_counts"]
    for class_id, count in train_counts.items():
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        # Cap bar length at 30 characters so it fits in an 80-col terminal.
        bar = "█" * min(count // 2, 30)
        print(f"      {class_id}: {class_name:20} {count:4} {bar}")

    return all_stats


def visualize_samples(num_samples: int = 6) -> None:
    """
    Create and save a grid of training samples with ground-truth overlays.

    Randomly selects ``num_samples`` image–label pairs from the training
    split, renders the segmentation annotations via
    :func:`visualize_segmentation`, and saves the resulting 2×3 grid to
    ``results/step0_samples.png``.

    Parameters
    ----------
    num_samples : int, optional
        Number of samples to display.  Capped at the available training
        image count.  Default is ``6`` (fills a 2×3 grid).
    """
    print("\n" + "=" * 60)
    print("VISUALIZING SAMPLES")
    print("=" * 60)

    pairs = get_image_label_pairs(TRAIN_PATH)
    if not pairs:
        print("  ❌ No image-label pairs found!")
        return

    # Draw without replacement; cap at the available pool size.
    selected = random.sample(pairs, min(num_samples, len(pairs)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()   # Treat as a flat list for indexed access.

    for idx, (img_path, label_path) in enumerate(selected):
        image       = cv2.imread(img_path)
        h, w        = image.shape[:2]
        objects     = parse_yolo_segmentation_label(label_path, w, h)
        vis_image   = visualize_segmentation(image, objects)

        axes[idx].imshow(vis_image)
        axes[idx].set_title(f"{os.path.basename(img_path)}\n{len(objects)} object(s)")
        axes[idx].axis('off')

    # Hide any subplot panels that weren't populated (e.g. < 6 samples available).
    for idx in range(len(selected), 6):
        axes[idx].axis('off')

    plt.suptitle("Sample Segmentation Visualizations", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "step0_samples.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  ✅ Saved visualization to: {output_path}")


def run_dataset_check() -> bool:
    """
    Orchestrate the full dataset check: structure → statistics → visualisation.

    This is the public entry point called by ``run_pipeline.py``
    (Step 0).  It exits early with ``False`` if the directory structure is
    invalid so the pipeline can abort before wasting time on training.

    Returns
    -------
    bool
        ``True`` when all checks pass and the dataset is ready for training.
        ``False`` when the directory structure is invalid.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " STEP 0: DATASET CHECK & STATISTICS ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    # 1. Verify directory layout.
    structure_ok = check_dataset_structure()
    if not structure_ok:
        print("\n  ❌ Dataset structure issues found. Please fix before training.")
        return False

    # 2. Compute and display dataset statistics.
    stats = get_dataset_statistics()

    # 3. Save sample visualisations for a visual sanity check.
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
