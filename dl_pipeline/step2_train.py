"""
=============================================================================
STEP 2: TRAIN YOLOV8 SEGMENTATION MODEL
=============================================================================

Fine-tune a pre-trained YOLOv8 nano segmentation model on the ISIC
skin-lesion dataset.  The model learns to simultaneously detect the bounding
box *and* pixel-level segmentation mask for each lesion.

What this module does
---------------------
``train_model``
    1. Auto-detects the best compute device (CUDA / CPU).
    2. Loads the ``data.yaml`` created by Step 1.
    3. Downloads / loads the chosen pre-trained YOLOv8 checkpoint.
    4. Calls ``model.train(…)`` with dermatology-specific augmentations
       (hue shifts for skin tone variation, 180° rotation because lesions
       appear at arbitrary angles, copy-paste to synthesise rare classes, …).
    5. Reports the best-model path and final validation metrics.

Augmentation rationale
----------------------
* ``hsv_h=0.015`` / ``hsv_s=0.7`` / ``hsv_v=0.4``
    Simulate patient-to-patient skin-tone and lighting variation.
* ``degrees=180``
    Lesions can be imaged at any orientation; full rotation prevents
    the model from learning spurious orientation biases.
* ``copy_paste=0.3``
    Copies lesion instances from one image onto another, artificially
    increasing the frequency of minority classes (e.g. DF, VASC).
* ``erasing=0.4``
    Simulates occluding objects like hair or ruler markers that are
    common in dermoscopy images.

Command-line usage
------------------
    # Train with config defaults
    python step2_train.py

    # Override epoch count
    python step2_train.py --epochs 100

    # Resume from the last checkpoint
    python step2_train.py --resume
"""

import os
import sys
import argparse
from typing import Optional

# Make the parent package importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl_pipeline.config import (
    MODEL_NAME, EPOCHS, BATCH_SIZE, IMAGE_SIZE,
    LEARNING_RATE, PATIENCE, WORKERS, OUTPUT_DIR,
    get_data_yaml_path, get_device, print_config
)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(
    epochs: Optional[int] = None,
    resume: bool = False,
) -> Optional[str]:
    """
    Fine-tune the YOLOv8 segmentation model on the skin-lesion dataset.

    Loads a pre-trained checkpoint (downloaded automatically by Ultralytics
    if not already cached), attaches the dataset via ``data.yaml``, and
    runs the full training loop with skin-lesion-specific augmentations.

    Training artefacts (weights, loss curves, confusion matrix, …) are
    saved under ``<OUTPUT_DIR>/yolov8_seg/``.

    Parameters
    ----------
    epochs : int, optional
        Number of training epochs.  Falls back to ``config.EPOCHS`` (50)
        when ``None``.
    resume : bool, optional
        When ``True``, continue training from the last saved checkpoint
        instead of starting from the pre-trained weights.
        Default is ``False``.

    Returns
    -------
    str or None
        Absolute path to the best model weights (``best.pt``), or
        ``None`` if a prerequisite check fails (e.g. missing ``data.yaml``).

    Raises
    ------
    RuntimeError
        Propagated from Ultralytics if the training loop itself fails
        (e.g. CUDA out-of-memory).  Reduce ``BATCH_SIZE`` in ``config.py``
        if this occurs on a low-VRAM GPU.
    """
    from ultralytics import YOLO

    # Display step header.
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " STEP 2: TRAIN YOLOV8 SEGMENTATION MODEL ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    # Detect and print the available compute device.
    device = get_device()

    # Echo the full configuration so the user can confirm settings before
    # a potentially long training run.
    print_config()

    # Allow the caller to override the epoch count; otherwise use the
    # module-level default from config.py.
    num_epochs = epochs if epochs is not None else EPOCHS

    # Ensure data.yaml was created by Step 1 before starting training.
    data_yaml = get_data_yaml_path()
    if not os.path.exists(data_yaml):
        print(f"  ❌ data.yaml not found at: {data_yaml}")
        print("     Run step1_prepare_data.py first!")
        return None

    print(f"  📄 Using data config: {data_yaml}")

    # Load the pre-trained YOLOv8 nano segmentation model.
    # Ultralytics fetches the .pt file from its servers on first use.
    print(f"\n  ⏳ Loading pretrained model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    print("  ✅ Model loaded!")

    # Announce the key training parameters before handing off to Ultralytics.
    print(f"\n  🚀 Starting training...")
    print(f"     Epochs: {num_epochs}")
    print(f"     Batch size: {BATCH_SIZE}")
    print(f"     Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"     Device: {device}")
    print("\n" + "=" * 60)

    # -----------------------------------------------------------------------
    # Launch training.
    # Key augmentation parameters are explained in the module docstring.
    # -----------------------------------------------------------------------
    results = model.train(
        data=data_yaml,           # Path to data.yaml (dataset definition)
        epochs=num_epochs,        # Total training epochs
        batch=BATCH_SIZE,         # Images per gradient update
        imgsz=IMAGE_SIZE,         # Input resolution (square)
        lr0=LEARNING_RATE,        # Initial learning rate
        patience=PATIENCE,        # Early-stopping patience (epochs)
        workers=WORKERS,          # DataLoader parallel workers
        device=device,            # 'cuda' or 'cpu'
        project=OUTPUT_DIR,       # Root output directory
        name="yolov8_seg",        # Sub-directory name for this run
        exist_ok=True,            # Overwrite previous run's output
        pretrained=True,          # Start from ImageNet pre-trained weights
        verbose=True,             # Print per-epoch metrics to stdout
        plots=True,               # Save training-curve and PR-curve plots
        save=True,                # Checkpoint weights each epoch
        resume=resume,            # Continue from last.pt if True
        # --- Dermatology-specific augmentations ----------------------------
        augment=True,
        hsv_h=0.015,              # Hue jitter    – skin tone variation
        hsv_s=0.7,                # Saturation    – lesion colour contrast
        hsv_v=0.4,                # Value/brightness variation
        degrees=180,              # Full rotation – lesions are orientation-agnostic
        fliplr=0.5,               # Horizontal flip probability
        flipud=0.5,               # Vertical flip probability
        scale=0.5,                # Random zoom range
        mosaic=1.0,               # Mosaic augmentation (combines 4 images)
        mixup=0.1,                # MixUp: blend pairs of images
        copy_paste=0.3,           # Copy-paste: transplant lesions across images
        erasing=0.4,              # Random erasing – simulates hair / occlusion
    )

    print("\n" + "=" * 60)
    print("  🎉 TRAINING COMPLETE!")
    print("=" * 60)

    # Build the expected paths to the saved checkpoints.
    best_model = os.path.join(OUTPUT_DIR, "yolov8_seg", "weights", "best.pt")
    last_model = os.path.join(OUTPUT_DIR, "yolov8_seg", "weights", "last.pt")

    print(f"\n  📁 Saved Models:")
    print(f"     Best: {best_model}")
    print(f"     Last: {last_model}")

    # Print any scalar metrics from the final training epoch.
    if hasattr(results, 'results_dict'):
        print(f"\n  📊 Final Metrics:")
        for key, value in results.results_dict.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("✅ STEP 2 COMPLETE!")
    print(f"   Best model saved to: {best_model}")
    print("=" * 60 + "\n")

    return best_model


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def run_training() -> Optional[str]:
    """
    Parse command-line arguments and start model training.

    Wraps :func:`train_model` with an ``argparse`` interface for direct
    invocation of this script.

    CLI flags
    ---------
    --epochs int   Number of epochs (overrides ``config.EPOCHS``).
    --resume       Resume training from the last saved checkpoint.

    Returns
    -------
    str or None
        The return value of :func:`train_model`.
    """
    parser = argparse.ArgumentParser(description="Train YOLOv8 Segmentation Model")
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    args = parser.parse_args()

    return train_model(epochs=args.epochs, resume=args.resume)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_training()
