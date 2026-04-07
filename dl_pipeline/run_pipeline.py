"""
=============================================================================
RUN COMPLETE DL PIPELINE
=============================================================================

Main entry point for orchestrating the full YOLOv8 deep-learning pipeline
for skin lesion segmentation.  Each pipeline step is encapsulated in its
own module; this script imports and calls them in order.

Pipeline overview
-----------------
    Step 0 – Dataset Check : verify folder structure, compute statistics,
                             save sample visualisations.
    Step 1 – Prepare Data  : write ``data.yaml`` that YOLOv8 requires.
    Step 2 – Train Model   : fine-tune YOLOv8-seg on the lesion dataset.
    Step 3 – Validate      : run the trained model on validation images and
                             report mAP metrics.
    Step 4 – Inference     : (separate script) run on arbitrary new images.

Command-line usage
------------------
    # Run the complete pipeline with config defaults
    python run_pipeline.py

    # Run a single step (0–3)
    python run_pipeline.py --step 0      # Dataset check only
    python run_pipeline.py --step 1      # Prepare data only
    python run_pipeline.py --step 2      # Training only
    python run_pipeline.py --step 3      # Validation only

    # Quick smoke-test (2 epochs)
    python run_pipeline.py --quick

    # Override epoch count
    python run_pipeline.py --epochs 100
"""

import os
import sys
import argparse
from typing import Optional

# Make the parent package importable when this script is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl_pipeline.config import print_config, ensure_output_dir


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def print_banner() -> None:
    """
    Print the ASCII-art pipeline banner to stdout.

    Shows the pipeline name and a numbered list of steps so the user
    immediately knows what the full run will do.
    """
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


# ============================================================================
# STEP RUNNERS
# Each function is a thin wrapper that lazily imports its step module so
# that heavy dependencies (e.g. ultralytics) are only loaded when needed.
# ============================================================================

def run_step0() -> bool:
    """
    Execute Step 0: dataset structure check and sample visualisation.

    Imports and calls ``step0_dataset_check.run_dataset_check``.

    Returns
    -------
    bool
        ``True`` when the dataset passes all structural checks,
        ``False`` if any required directory is missing.
    """
    from dl_pipeline.step0_dataset_check import run_dataset_check
    return run_dataset_check()


def run_step1() -> str:
    """
    Execute Step 1: generate the ``data.yaml`` file required by YOLOv8.

    Imports and calls ``step1_prepare_data.run_prepare_data``.

    Returns
    -------
    str
        Absolute path to the newly created ``data.yaml`` file.
    """
    from dl_pipeline.step1_prepare_data import run_prepare_data
    return run_prepare_data()


def run_step2(epochs: Optional[int] = None) -> Optional[str]:
    """
    Execute Step 2: train the YOLOv8 segmentation model.

    Imports and calls ``step2_train.train_model``.

    Parameters
    ----------
    epochs : int, optional
        Number of training epochs.  When ``None``, the value from
        ``config.EPOCHS`` is used.

    Returns
    -------
    str or None
        Absolute path to the best model weights (``best.pt``), or
        ``None`` if training failed.
    """
    from dl_pipeline.step2_train import train_model
    return train_model(epochs=epochs)


def run_step3(model_path: Optional[str] = None):
    """
    Execute Step 3: validate the trained model and compute metrics.

    Imports and calls ``step3_validate.run_validation``.

    Parameters
    ----------
    model_path : str, optional
        Path to model weights.  Defaults to the ``best.pt`` produced by
        Step 2 when not provided.

    Returns
    -------
    ultralytics.utils.metrics.DetMetrics or None
        Validation metrics object, or ``None`` if loading the model failed.
    """
    from dl_pipeline.step3_validate import run_validation
    return run_validation(model_path=model_path)


# ============================================================================
# FULL-PIPELINE ORCHESTRATOR
# ============================================================================

def run_full_pipeline(
    epochs: Optional[int] = None,
    quick: bool = False,
) -> Optional[str]:
    """
    Run the complete pipeline from Step 0 through Step 3.

    Executes each step in sequence, stopping early if a critical step fails
    (e.g. dataset check).  Prints a rich summary with next-step guidance
    upon successful completion.

    Parameters
    ----------
    epochs : int, optional
        Override number of training epochs.  Ignored when ``quick=True``.
    quick : bool, optional
        When ``True``, forces ``epochs=2`` for a rapid smoke-test run.
        Default is ``False``.

    Returns
    -------
    str or None
        Absolute path to the best trained model weights, or ``None`` if
        the pipeline was aborted due to an error.
    """
    print_banner()
    print_config()
    ensure_output_dir()

    # Quick mode overrides any user-supplied epoch count.
    if quick:
        epochs = 2
        print("  ⚡ QUICK MODE: Using 2 epochs for testing\n")

    # ------------------------------------------------------------------
    # Step 0 – Dataset Check
    # ------------------------------------------------------------------
    print("\n" + "█" * 60)
    print("  RUNNING STEP 0: DATASET CHECK")
    print("█" * 60)

    if not run_step0():
        # A failed dataset check means training cannot proceed.
        print("\n  ❌ Dataset check failed. Please fix issues before training.")
        return None

    # ------------------------------------------------------------------
    # Step 1 – Prepare Data
    # ------------------------------------------------------------------
    print("\n" + "█" * 60)
    print("  RUNNING STEP 1: PREPARE DATA")
    print("█" * 60)

    run_step1()

    # ------------------------------------------------------------------
    # Step 2 – Train Model
    # ------------------------------------------------------------------
    print("\n" + "█" * 60)
    print("  RUNNING STEP 2: TRAIN MODEL")
    print("█" * 60)

    model_path = run_step2(epochs=epochs)

    if model_path is None:
        print("\n  ❌ Training failed.")
        return None

    # ------------------------------------------------------------------
    # Step 3 – Validate Model
    # ------------------------------------------------------------------
    print("\n" + "█" * 60)
    print("  RUNNING STEP 3: VALIDATE MODEL")
    print("█" * 60)

    run_step3(model_path=model_path)

    # ------------------------------------------------------------------
    # Success summary
    # ------------------------------------------------------------------
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


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Parse command-line arguments and dispatch to the appropriate runner.

    When ``--step`` is supplied, only that step is executed.
    Otherwise the full pipeline (Steps 0–3) is run in sequence.

    CLI flags
    ---------
    --step  {0,1,2,3}   Run only the specified step.
    --epochs int         Number of training epochs (Step 2 only).
    --quick              Use 2 epochs for a fast smoke-test.
    --model str          Path to model weights (Step 3 only).
    """
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
        # Single-step mode: run only the requested step.
        print_banner()

        if args.step == 0:
            run_step0()
        elif args.step == 1:
            run_step1()
        elif args.step == 2:
            # Honour --quick flag even in single-step mode.
            epochs = 2 if args.quick else args.epochs
            run_step2(epochs=epochs)
        elif args.step == 3:
            run_step3(model_path=args.model)
    else:
        # Full-pipeline mode.
        epochs = 2 if args.quick else args.epochs
        run_full_pipeline(epochs=epochs, quick=args.quick)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
