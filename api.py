import os
import io
import json
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np

# Try to import from config, otherwise define fallbacks
try:
    from dl_pipeline.config import CLASS_NAMES, CLASS_SEVERITY, CLASS_CODES, CONF_THRESHOLD, IOU_THRESHOLD
except ImportError:
    CLASS_NAMES = {
        0: "Benign Keratosis",
        1: "Melanocytic Nevi",
        2: "Dermatofibroma",
        3: "Melanoma",
        4: "Vascular Lesion",
        5: "Basal Cell Carcinoma",
        6: "Actinic Keratoses",
    }
    CLASS_SEVERITY = {
        0: 0, 1: 0, 2: 0, 3: 2, 4: 0, 5: 2, 6: 1
    }
    CLASS_CODES = {
        0: "BKL", 1: "NV", 2: "DF", 3: "MEL", 4: "VASC", 5: "BCC", 6: "AKIEC"
    }
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45

# Severity → BGR color mapping
SEVERITY_COLORS = {
    0: (0, 255, 0),     # Green  – Benign
    1: (0, 165, 255),   # Orange – Pre-cancerous
    2: (0, 0, 255),     # Red    – Malignant
}

app = FastAPI(title="DermaDoc YOLOv8 API", description="API to test skin lesion segmentation model.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "dl_pipeline", "results", "yolov8_seg", "weights", "best.pt")

if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model not found at {MODEL_PATH}. Using yolov8n-seg.pt as fallback.")
    model = YOLO('yolov8n-seg.pt')
else:
    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "DermaDoc YOLOv8 API is running. Send a POST request to /predict with an image file."}

@app.get("/health")
def health():
    try:
        model_loaded = model is not None and hasattr(model, "model_name")
    except Exception:
        model_loaded = False
    return {
        "status": "ready" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        h, w = img.shape[:2]

        # Run prediction with segmentation
        results = model.predict(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        detections = []
        vis_image = img.copy()
        overlay = vis_image.copy()

        for result in results:
            boxes = result.boxes
            masks = result.masks  # Segmentation masks

            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                xyxy = box.xyxy.cpu().numpy()[0].tolist()

                class_name = CLASS_NAMES.get(class_id, "Unknown")
                class_code = CLASS_CODES.get(class_id, f"C{class_id}")
                severity = CLASS_SEVERITY.get(class_id, 0)
                color = SEVERITY_COLORS.get(severity, (0, 255, 0))

                # ── Draw segmentation mask if available ──
                mask_polygon = []
                if masks is not None and i < len(masks.data):
                    mask = masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_bool = mask_resized > 0.5

                    # Fill mask on overlay
                    overlay[mask_bool] = color

                    # Draw contour outline
                    contours, _ = cv2.findContours(
                        mask_bool.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(vis_image, contours, -1, color, 2)

                    # Extract contour points for frontend (optional)
                    for contour in contours:
                        points = contour.squeeze().tolist()
                        if isinstance(points[0], list):
                            mask_polygon.extend(points)

                # ── Draw label ──
                severity_text = ["Benign", "Pre-cancerous", "Malignant"][severity]
                label = f"{class_code} {severity_text} {confidence:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                lx, ly = int(xyxy[0]), max(25, int(xyxy[1]) - 8)
                cv2.rectangle(vis_image, (lx, ly - th - 6), (lx + tw + 6, ly + 4), color, -1)
                cv2.putText(vis_image, label, (lx + 3, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                detections.append({
                    "class_id": class_id,
                    "class_code": class_code,
                    "class_name": class_name,
                    "severity": severity,
                    "severity_label": severity_text,
                    "confidence": round(confidence, 4),
                    "bbox": [round(x, 2) for x in xyxy],
                })

        # Blend segmentation overlay onto image
        vis_image = cv2.addWeighted(overlay, 0.35, vis_image, 0.65, 0)

        # Encode annotated image as base64 JPEG
        _, buffer = cv2.imencode('.jpg', vis_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={
            "status": "success",
            "num_detections": len(detections),
            "detections": detections,
            "image_base64": image_base64,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
