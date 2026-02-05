from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import logging
import uuid

from app.config import UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from app.models.lstm_model import detector
from app.services.audio_processor import AudioProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize processor
processor = AudioProcessor()

@router.get("/")
async def root():
    """API info endpoint"""
    return {
        "message": "Deepfake Audio Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict (POST)",
        }
    }

@router.get("/health")
async def health_check():
    """Check API and model health"""
    model_status = detector.health_check()
    return {
        "status": "healthy" if model_status["model_loaded"] else "unhealthy",
        "model": model_status
    }

@router.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Audio file (wav, mp3, flac, ogg)"),
    threshold: float = Form(0.3, description="Detection threshold (0.0-1.0)")
):
    """
    Upload audio file and detect if it's deepfake
    
    - **file**: Audio file to analyze
    - **threshold**: Decision threshold (default 0.3, lower = more sensitive to fakes)
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    safe_filename = f"{file_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            file_path.unlink()
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
        # Get audio info
        audio_info = processor.get_audio_info(file_path)
        
        # Process and predict
        logger.info(f"Processing {file.filename}...")
        features = processor.process_file(file_path)
        
        logger.info(f"Running prediction (threshold={threshold})...")
        result = detector.predict(features, threshold=threshold)
        
        # Add metadata to result
        result.update({
            "filename": file.filename,
            "file_size_bytes": file_size,
            "audio_info": audio_info,
            "features_shape": list(features.shape)
        })
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Cleanup
        if file_path.exists():
            file_path.unlink(missing_ok=True)