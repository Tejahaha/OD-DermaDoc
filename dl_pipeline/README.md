# DL Pipeline - YOLOv8 Skin Lesion Segmentation

A modular deep learning pipeline for skin lesion detection and segmentation using YOLOv8.

## Quick Start

```powershell
# Run the full pipeline
python dl_pipeline/run_pipeline.py

# Quick test (2 epochs)
python dl_pipeline/run_pipeline.py --quick

# Train for 100 epochs
python dl_pipeline/run_pipeline.py --epochs 100
```

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 0 | `step0_dataset_check.py` | Verify dataset structure & visualize samples |
| 1 | `step1_prepare_data.py` | Create `data.yaml` for YOLOv8 |
| 2 | `step2_train.py` | Train YOLOv8 segmentation model |
| 3 | `step3_validate.py` | Evaluate & visualize predictions |
| 4 | `step4_inference.py` | Run inference on new images |

## Running Individual Steps

```powershell
# Dataset check only
python dl_pipeline/run_pipeline.py --step 0

# Training only
python dl_pipeline/run_pipeline.py --step 2

# Validate with custom model
python dl_pipeline/step3_validate.py --model path/to/model.pt
```

## Inference on New Images

```powershell
# Single image
python dl_pipeline/step4_inference.py --image path/to/image.jpg

# Folder of images
python dl_pipeline/step4_inference.py --folder path/to/images/

# Custom output location
python dl_pipeline/step4_inference.py --image test.jpg --output results/
```

## Configuration

Edit `config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 50 | Training epochs |
| `BATCH_SIZE` | 8 | Batch size (reduce if OOM) |
| `IMAGE_SIZE` | 640 | Input image size |
| `MODEL_SIZE` | 'n' | Model size: n/s/m/l/x |
| `CONF_THRESHOLD` | 0.25 | Detection confidence |

## Hardware Requirements

- **GPU**: NVIDIA RTX with CUDA (8GB+ VRAM recommended)
- **RAM**: 16GB+
- **Disk**: ~5GB for dataset + models

## Output Structure

```
dl_pipeline/
├── results/
│   ├── yolov8_seg/
│   │   ├── weights/
│   │   │   ├── best.pt      # Best model
│   │   │   └── last.pt      # Last checkpoint
│   │   └── ...              # Training plots
│   ├── step0_samples.png    # Dataset samples
│   └── step3_validation.png # Validation results
```

## Class Labels

| ID | Code | Name | Severity |
|----|------|------|----------|
| 0 | BKL | Benign Keratosis | Benign |
| 1 | NV | Melanocytic Nevi | Benign |
| 2 | DF | Dermatofibroma | Benign |
| 3 | MEL | Melanoma | ⚠️ Malignant |
| 4 | VASC | Vascular Lesion | Benign |
| 5 | BCC | Basal Cell Carcinoma | ⚠️ Malignant |
| 6 | AKIEC | Actinic Keratoses | Pre-cancerous |
