"""
=============================================================================
STEP 2: TRAIN YOLOV8 SEGMENTATION MODEL
=============================================================================

Train YOLOv8 on your skin lesion dataset for instance segmentation.
The model learns to detect and segment skin lesions in images.

Usage:
    python step2_train.py
    
    # Or with custom epochs:
    python step2_train.py --epochs 100
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl_pipeline.config import (
    MODEL_NAME, EPOCHS, BATCH_SIZE, IMAGE_SIZE,
    LEARNING_RATE, PATIENCE, WORKERS, OUTPUT_DIR,
    get_data_yaml_path, get_device, print_config
)


def train_model(epochs=None, resume=False):
    """
    Train the YOLOv8 segmentation model.
    
    Args:
        epochs: Number of training epochs (uses config default if None)
        resume: Resume from last checkpoint if True
    
    Returns:
        Path to the best model weights
    """
    from ultralytics import YOLO
    
    # Print banner
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " STEP 2: TRAIN YOLOV8 SEGMENTATION MODEL ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Check GPU
    device = get_device()
    
    # Print configuration
    print_config()
    
    # Use provided epochs or default from config
    num_epochs = epochs if epochs is not None else EPOCHS
    
    # Check data.yaml exists
    data_yaml = get_data_yaml_path()
    if not os.path.exists(data_yaml):
        print(f"  ❌ data.yaml not found at: {data_yaml}")
        print("     Run step1_prepare_data.py first!")
        return None
    
    print(f"  📄 Using data config: {data_yaml}")
    
    # Load pretrained model
    print(f"\n  ⏳ Loading pretrained model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    print("  ✅ Model loaded!")
    
    # Training configuration
    print(f"\n  🚀 Starting training...")
    print(f"     Epochs: {num_epochs}")
    print(f"     Batch size: {BATCH_SIZE}")
    print(f"     Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"     Device: {device}")
    print("\n" + "=" * 60)
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=num_epochs,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        lr0=LEARNING_RATE,
        patience=PATIENCE,
        workers=WORKERS,
        device=device,
        project=OUTPUT_DIR,
        name="yolov8_seg",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=True,
        save=True,
        resume=resume,
    )
    
    print("\n" + "=" * 60)
    print("  🎉 TRAINING COMPLETE!")
    print("=" * 60)
    
    # Get model paths
    best_model = os.path.join(OUTPUT_DIR, "yolov8_seg", "weights", "best.pt")
    last_model = os.path.join(OUTPUT_DIR, "yolov8_seg", "weights", "last.pt")
    
    print(f"\n  📁 Saved Models:")
    print(f"     Best: {best_model}")
    print(f"     Last: {last_model}")
    
    # Print training results
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


def run_training():
    """Run training with command line arguments."""
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
