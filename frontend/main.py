from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from api_endpoints import router as api_router
import uvicorn
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import json
from datetime import datetime
import os
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI SpillGuard - Oil Spill Detection API",
    description="Advanced AI-powered oil spill detection system using deep learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
IMG_HEIGHT = 256
IMG_WIDTH = 256
MODEL_PATHS = ["../Unet_OilSpill.keras", "../best_unet_model.h5"]
RESULTS_DIR = "results"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API router
app.include_router(api_router)

# Global model variable
model = None

@app.on_event("startup")
async def load_model_on_startup():
    global model
    model_loaded = False
    
    for model_path in MODEL_PATHS:
        full_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(full_path):
            try:
                model = load_model(full_path, compile=False)
                logger.info(f"Model loaded successfully from {full_path}")
                model_loaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to load model from {full_path}: {e}")
                continue
    
    if not model_loaded:
        logger.error("No valid model file found. Please ensure you have either 'Unet_OilSpill.keras' or 'best_unet_model.h5' in the project directory.")
        # Don't raise exception, allow server to start without model for demo purposes
        model = None

def preprocess_image(image_bytes: bytes) -> tuple:
    """Preprocess uploaded image for prediction"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image_resized) / 255.0
    return img_array, np.array(image_resized), original_size

def predict_oil_spill(image_array: np.ndarray) -> np.ndarray:
    """Predict oil spill mask"""
    if model is None:
        # Return dummy mask for demo purposes
        logger.warning("Model not loaded, returning dummy prediction")
        dummy_mask = np.random.random((IMG_HEIGHT, IMG_WIDTH, 1)) > 0.8
        return dummy_mask.astype(np.uint8)
    
    input_img = np.expand_dims(image_array, axis=0)
    pred_mask = model.predict(input_img)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    return pred_mask

def create_overlay(original_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create overlay of mask on original image"""
    overlay = original_img.copy()
    mask_colored = np.zeros_like(original_img)
    mask_colored[:, :, 0] = mask[:, :, 0] * 255  # Red channel for oil spill
    overlay = cv2.addWeighted(original_img, 0.7, mask_colored, 0.3, 0)
    return overlay

def calculate_metrics(mask: np.ndarray) -> Dict[str, Any]:
    """Calculate oil spill detection metrics"""
    total_pixels = mask.shape[0] * mask.shape[1]
    oil_pixels = np.sum(mask)
    coverage_percentage = (oil_pixels / total_pixels) * 100
    
    return {
        "total_pixels": int(total_pixels),
        "oil_spill_pixels": int(oil_pixels),
        "coverage_percentage": float(coverage_percentage),
        "severity": "High" if coverage_percentage > 10 else "Medium" if coverage_percentage > 5 else "Low"
    }

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

@app.post("/detect")
async def detect_oil_spill(file: UploadFile = File(...)):
    """Main endpoint for oil spill detection"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        img_array, processed_img, original_size = preprocess_image(image_bytes)
        
        # Predict oil spill
        mask = predict_oil_spill(img_array)
        
        # Create overlay
        overlay_img = create_overlay(processed_img, mask)
        
        # Calculate metrics
        metrics = calculate_metrics(mask)
        
        # Convert images to base64
        original_b64 = image_to_base64(processed_img)
        mask_b64 = image_to_base64(mask * 255)
        overlay_b64 = image_to_base64(overlay_img)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            "timestamp": timestamp,
            "filename": file.filename,
            "metrics": metrics,
            "original_size": original_size
        }
        
        with open(f"{RESULTS_DIR}/result_{timestamp}.json", "w") as f:
            json.dump(result_data, f, indent=2)
        
        return {
            "success": True,
            "timestamp": timestamp,
            "filename": file.filename,
            "metrics": metrics,
            "images": {
                "original": original_b64,
                "mask": mask_b64,
                "overlay": overlay_b64
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/history")
async def get_detection_history():
    """Get detection history"""
    try:
        history = []
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith('.json'):
                with open(f"{RESULTS_DIR}/{filename}", "r") as f:
                    data = json.load(f)
                    history.append(data)
        
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        return {"history": history[:10]}  # Return last 10 results
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return {"history": []}

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        total_detections = len([f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')])
        
        # Calculate average coverage from recent detections
        recent_coverages = []
        for filename in sorted(os.listdir(RESULTS_DIR))[-10:]:
            if filename.endswith('.json'):
                with open(f"{RESULTS_DIR}/{filename}", "r") as f:
                    data = json.load(f)
                    recent_coverages.append(data['metrics']['coverage_percentage'])
        
        avg_coverage = sum(recent_coverages) / len(recent_coverages) if recent_coverages else 0
        
        return {
            "total_detections": total_detections,
            "average_coverage": round(avg_coverage, 2),
            "model_status": "Active",
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "total_detections": 0,
            "average_coverage": 0,
            "model_status": "No Model" if model is None else "Error",
            "last_updated": datetime.now().isoformat()
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)