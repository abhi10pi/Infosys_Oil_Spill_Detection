from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from datetime import datetime

router = APIRouter(prefix="/api/v1", tags=["Oil Spill Detection API"])

class DetectionResult(BaseModel):
    timestamp: str
    filename: str
    coverage_percentage: float
    severity: str
    oil_spill_pixels: int
    total_pixels: int

class SystemHealth(BaseModel):
    status: str
    ai_model_loaded: bool
    total_detections: int
    uptime: str
    memory_usage: Optional[str] = None

class AlertConfig(BaseModel):
    email: Optional[str] = None
    threshold: float = 5.0
    enabled: bool = False

@router.get("/health", response_model=SystemHealth)
async def get_system_health():
    """Get system health status"""
    try:
        results_dir = "results"
        total_detections = len([f for f in os.listdir(results_dir) if f.endswith('.json')]) if os.path.exists(results_dir) else 0
        
        return SystemHealth(
            status="healthy",
            ai_model_loaded=True,
            total_detections=total_detections,
            uptime="Running",
            memory_usage="Normal"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/detections", response_model=List[DetectionResult])
async def get_all_detections(limit: int = 50, offset: int = 0):
    """Get all detection results with pagination"""
    try:
        results_dir = "results"
        if not os.path.exists(results_dir):
            return []
        
        detections = []
        files = sorted([f for f in os.listdir(results_dir) if f.endswith('.json')], reverse=True)
        
        for filename in files[offset:offset+limit]:
            with open(f"{results_dir}/{filename}", "r") as f:
                data = json.load(f)
                detections.append(DetectionResult(
                    timestamp=data['timestamp'],
                    filename=data['filename'],
                    coverage_percentage=data['metrics']['coverage_percentage'],
                    severity=data['metrics']['severity'],
                    oil_spill_pixels=data['metrics']['oil_spill_pixels'],
                    total_pixels=data['metrics']['total_pixels']
                ))
        
        return detections
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving detections: {str(e)}")

@router.get("/detections/{detection_id}")
async def get_detection_details(detection_id: str):
    """Get detailed information about a specific detection"""
    try:
        results_dir = "results"
        file_path = f"{results_dir}/result_{detection_id}.json"
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Detection not found")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        return data
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Detection not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving detection: {str(e)}")

@router.delete("/detections/{detection_id}")
async def delete_detection(detection_id: str):
    """Delete a specific detection result"""
    try:
        results_dir = "results"
        file_path = f"{results_dir}/result_{detection_id}.json"
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Detection not found")
        
        os.remove(file_path)
        return {"message": "Detection deleted successfully"}
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Detection not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting detection: {str(e)}")

@router.get("/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary of all detections"""
    try:
        results_dir = "results"
        if not os.path.exists(results_dir):
            return {
                "total_detections": 0,
                "average_coverage": 0,
                "severity_distribution": {"High": 0, "Medium": 0, "Low": 0},
                "monthly_trend": []
            }
        
        detections = []
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(f"{results_dir}/{filename}", "r") as f:
                    data = json.load(f)
                    detections.append(data)
        
        if not detections:
            return {
                "total_detections": 0,
                "average_coverage": 0,
                "severity_distribution": {"High": 0, "Medium": 0, "Low": 0},
                "monthly_trend": []
            }
        
        # Calculate analytics
        total_detections = len(detections)
        average_coverage = sum(d['metrics']['coverage_percentage'] for d in detections) / total_detections
        
        severity_distribution = {"High": 0, "Medium": 0, "Low": 0}
        for detection in detections:
            severity = detection['metrics']['severity']
            if severity in severity_distribution:
                severity_distribution[severity] += 1
        
        return {
            "total_detections": total_detections,
            "average_coverage": round(average_coverage, 2),
            "severity_distribution": severity_distribution,
            "monthly_trend": []  # Could be implemented with more complex date parsing
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")

@router.post("/alerts/configure")
async def configure_alerts(config: AlertConfig):
    """Configure alert settings"""
    try:
        # Save alert configuration
        config_path = "alert_config.json"
        with open(config_path, "w") as f:
            json.dump(config.dict(), f, indent=2)
        
        return {"message": "Alert configuration saved successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving alert configuration: {str(e)}")

@router.get("/alerts/config")
async def get_alert_config():
    """Get current alert configuration"""
    try:
        config_path = "alert_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            return config
        else:
            return AlertConfig().dict()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving alert configuration: {str(e)}")