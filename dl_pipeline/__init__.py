"""
dl_pipeline
===========

YOLOv8-based deep learning pipeline for skin lesion segmentation.

This package provides a modular, step-by-step workflow:

    Step 0 – ``step0_dataset_check``  : verify dataset structure & visualise samples
    Step 1 – ``step1_prepare_data``   : generate ``data.yaml`` for YOLOv8
    Step 2 – ``step2_train``          : train the segmentation model
    Step 3 – ``step3_validate``       : evaluate model and visualise predictions
    Step 4 – ``step4_inference``      : run the trained model on new images

The ``config`` sub-module is the single source of truth for all paths,
hyperparameters, and class definitions used across every step.

Typical entry points
--------------------
    # Run the full pipeline end-to-end
    python dl_pipeline/run_pipeline.py

    # Run a single step
    python dl_pipeline/step2_train.py --epochs 50

    # Import the config inside another module
    from dl_pipeline import config
    from dl_pipeline.config import DATASET_PATH, CLASS_NAMES
"""

from . import config
