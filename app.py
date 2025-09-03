from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from typing import List
import logging
from contextlib import asynccontextmanager
import gdown   # <-- NEW: for downloading from Google Drive

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and classes
model = None
plant_classes = None

# Google Drive link (replace with your file ID)
MODEL_PATH = "plant_species_Model_kaggle.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=115K_QMpftnQxZ3DwoUargZp5WkOvGXcA"

def download_model():
    """Download model from Google Drive if not exists"""
    if not os.path.exists(MODEL_PATH):
        logger.info("Model not found locally. Downloading from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        logger.info("Model downloaded successfully.")

def _load_classes_from_data_dir(data_dir: str = "Data") -> List[str]:
    """Derive class names from subfolders inside the dataset directory, sorted alphabetically."""
    if not os.path.isdir(data_dir):
        logger.warning(f"Data directory not found at '{data_dir}'. Class names will be unavailable.")
        return []
    subfolders = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    subfolders.sort()
    if not subfolders:
        logger.warning(f"No class subfolders detected in '{data_dir}'. Class names will be unavailable.")
    return subfolders


def load_model():
    """Load the trained model and derive plant classes from Data/ directory if available."""
    global model, plant_classes

    try:
        # Ensure model exists (download if missing)
        download_model()

        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")

        # Derive classes from Data/ directory (no JSON required)
        plant_classes = _load_classes_from_data_dir("Data")
        if plant_classes:
            logger.info(f"Derived {len(plant_classes)} classes from 'Data' directory")
        else:
            logger.info("Proceeding without class names (predictions will return class indices only)")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Lifespan context manager to replace deprecated on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Plant Classification API...")
    load_model()
    logger.info("API startup complete")
    yield
    # Shutdown (if needed)
    logger.info("Shutting down Plant Classification API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Plant Species Classification API",
    description="A FastAPI service for classifying plant species using ResNet50",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for model prediction"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Plant Species Classification API",
        "version": "1.0.0",
        "status": "running",
        "num_classes": len(plant_classes) if plant_classes else 0
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_loaded": plant_classes is not None,
        "num_classes": len(plant_classes) if plant_classes else 0
    }

@app.get("/classes")
async def get_classes():
    if not plant_classes:
        return {"classes": [], "count": 0}
    return {"classes": plant_classes, "count": len(plant_classes)}

@app.post("/predict")
async def predict_plant(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        result = {
            "index": int(predicted_class_index),
            "confidence": confidence,
            "filename": file.filename
        }
        if plant_classes and 0 <= int(predicted_class_index) < len(plant_classes):
            result["predicted_class"] = plant_classes[int(predicted_class_index)]
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-base64")
async def predict_plant_base64(request: dict):
    try:
        if 'image' not in request:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        import base64
        image_data = base64.b64decode(request['image'])
        processed_image = preprocess_image(image_data)
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        result = {
            "index": int(predicted_class_index),
            "confidence": confidence
        }
        if plant_classes and 0 <= int(predicted_class_index) < len(plant_classes):
            result["predicted_class"] = plant_classes[int(predicted_class_index)]
        return result
    except Exception as e:
        logger.error(f"Base64 prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")
    results = []
    for file in files:
        try:
            if not file.content_type.startswith('image/'):
                results.append({"filename": file.filename, "error": "File must be an image"})
                continue
            image_bytes = await file.read()
            processed_image = preprocess_image(image_bytes)
            predictions = model.predict(processed_image, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])
            item = {
                "filename": file.filename,
                "index": int(predicted_class_index),
                "confidence": confidence
            }
            if plant_classes and 0 <= int(predicted_class_index) < len(plant_classes):
                item["predicted_class"] = plant_classes[int(predicted_class_index)]
            results.append(item)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
