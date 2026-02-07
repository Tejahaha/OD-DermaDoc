"""
=============================================================================
RUN COMPLETE DL PIPELINE
=============================================================================

This is the main entry point for running the complete YOLOv8 deep learning
pipeline for skin lesion segmentation.

Usage:
    # Run full pipeline
    python run_pipeline.py
    
    # Run specific steps
    python run_pipeline.py --step 0      # Dataset check only
    python run_pipeline.py --step 1      # Prepare data only
    python run_pipeline.py --step 2      # Training only
    python run_pipeline.py --step 3      # Validation only
    
    # Quick training (2 epochs for testing)
    python run_pipeline.py --quick
    
    # Custom epochs
    python run_pipeline.py --epochs 100
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl_pipeline.config import print_config, ensure_output_dir


def print_banner():
    """Print the pipeline banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   🔬 YOLOV8 SEGMENTATION PIPELINE FOR SKIN LESION ANALYSIS            ║
    ║                                                                       ║
    ║   A modular, GPU-accelerated deep learning pipeline                   ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝

    Pipeline Steps:
    ───────────────────────────────────────────────────────────────────────
    [Step 0] Dataset Check     - Verify structure & visualize samples
    [Step 1] Prepare Data      - Create data.yaml for YOLOv8
    [Step 2] Train Model       - Train YOLOv8 segmentation model
    [Step 3] Validate          - Evaluate & visualize predictions
    [Step 4] Inference         - Run on new images (separate script)
    ───────────────────────────────────────────────────────────────────────
    """
    print(banner)


def run_step0():
    """Run dataset check."""
    from dl_pipeline.step0_dataset_check import run_dataset_check
    return run_dataset_check()


def run_step1():
    """Run data preparation."""
    from dl_pipeline.step1_prepare_data import run_prepare_data
    return run_prepare_data()


def run_step2(epochs=None):
    """Run training."""
    from dl_pipeline.step2_train import train_model
    return train_model(epochs=epochs)


def run_step3(model_path=None):
    """Run validation."""
    from dl_pipeline.step3_validate import run_validation
    return run_validation(model_path=model_path)


def run_full_pipeline(epochs=None, quick=False):
    """
    Run the complete pipeline from start to finish.
    
    Args:
        epochs: Override number of training epochs
        quick: If True, use only 2 epochs for quick testing
    
    Returns:
        Path to the trained model
    """
    print_banner()
    print_config()
    ensure_output_dir()
    
    if quick:
        epochs = 2
        print("  ⚡ QUICK MODE: Using 2 epochs for testing\n")
    
    # Step 0: Dataset Check
    print("\n" + "█" * 60)
    print("  RUNNING STEP 0: DATASET CHECK")
    print("█" * 60)
    
    if not run_step0():
        print("\n  ❌ Dataset check failed. Please fix issues before training.")
        return None
    
    # Step 1: Prepare Data
    print("\n" + "█" * 60)
    print("  RUNNING STEP 1: PREPARE DATA")
    print("█" * 60)
    
    run_step1()
    
    # Step 2: Training
    print("\n" + "█" * 60)
    print("  RUNNING STEP 2: TRAIN MODEL")
    print("█" * 60)
    
    model_path = run_step2(epochs=epochs)
    
    if model_path is None:
        print("\n  ❌ Training failed.")
        return None
    
    # Step 3: Validation
    print("\n" + "█" * 60)
    print("  RUNNING STEP 3: VALIDATE MODEL")
    print("█" * 60)
    
    run_step3(model_path=model_path)
    
    # Complete!
    print("\n" + "═" * 60)
    print("  🎉 PIPELINE COMPLETE!")
    print("═" * 60)
    print(f"""
    ✅ Model trained and validated successfully!
    
    📁 Results saved to: {os.path.dirname(model_path)}
    
    📊 Next steps:
       1. Review validation results in the results folder
       2. Use step4_inference.py to run on new images:
          
          python dl_pipeline/step4_inference.py --image your_image.jpg
    
    💡 Tips:
       - For better accuracy, train with more epochs (50-100)
       - Check training curves for overfitting
       - Consider using a larger model (yolov8s-seg) if GPU allows
    """)
    
    return model_path


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Skin Lesion Segmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                 # Run full pipeline
  python run_pipeline.py --quick         # Quick test (2 epochs)
  python run_pipeline.py --epochs 100    # Train for 100 epochs
  python run_pipeline.py --step 0        # Dataset check only
  python run_pipeline.py --step 2        # Training only
        """
    )
    
    parser.add_argument('--step', type=int, choices=[0, 1, 2, 3],
                        help='Run specific step only (0-3)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 2 epochs')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model (for step 3)')
    
    args = parser.parse_args()
    
    if args.step is not None:
        # Run specific step
        print_banner()
        
        if args.step == 0:
            run_step0()
        elif args.step == 1:
            run_step1()
        elif args.step == 2:
            epochs = 2 if args.quick else args.epochs
            run_step2(epochs=epochs)
        elif args.step == 3:
            run_step3(model_path=args.model)
    else:
        # Run full pipeline
        epochs = 2 if args.quick else args.epochs
        run_full_pipeline(epochs=epochs, quick=args.quick)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
