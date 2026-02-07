"""
Traditional Machine Learning Pipeline for Skin Lesion Classification

This package provides a modular, step-by-step approach to image classification:

Modules:
    - config.py              : Configuration settings and paths
    - step1_load_data.py     : Load images and labels from dataset
    - step2_extract_features.py : Extract color, texture, shape features
    - step3_train_models.py  : Train and evaluate ML classifiers
    - step4_visualize.py     : Visualize results and insights
    - step5_inference.py     : Predict on new images
    - run_pipeline.py        : Main script to run the complete pipeline
"""

__version__ = "1.0.0"
__author__ = "DermaDoc"
