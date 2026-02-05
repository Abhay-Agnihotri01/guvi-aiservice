from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import base64
import logging
from datetime import datetime

from app.config import settings
from app.services.audio_processor import AudioProcessor
from app.services.ensemble_detector import EnsembleDetector
from app.services.explainer import ExplainabilityEngine
from app.utils.language_detector import LanguageDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="AI-Generated Voice Detection API",
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

# Initialize services
audio_processor = AudioProcessor()
ensemble_detector = EnsembleDetector()
explainability_engine = ExplainabilityEngine()
language_detector = LanguageDetector()

class AnalyzeRequest(BaseModel):
    audio_base64: str
    language_hint: Optional[str] = None

class AnalyzeResponse(BaseModel):
    classification: str
    confidence: float
    language: str
    explanation: Dict
    processing_time_ms: int

@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": ensemble_detector.models_loaded()
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(request: AnalyzeRequest):
    """
    Analyze audio for AI-generated voice detection
    
    Args:
        request: AnalyzeRequest with base64 encoded audio and optional language hint
    
    Returns:
        AnalyzeResponse with classification, confidence, and explanation
    """
    try:
        start_time = datetime.now()
        
        # Decode audio
        logger.info("Decoding audio from base64")
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Process audio
        logger.info("Processing audio")
        audio_array, sample_rate = audio_processor.process_audio(audio_bytes)
        
        # Detect language
        logger.info("Detecting language")
        detected_language = language_detector.detect(
            audio_array, 
            sample_rate, 
            language_hint=request.language_hint
        )
        
        # Run ensemble detection
        logger.info(f"Running ensemble detection for {detected_language}")
        detection_result = ensemble_detector.detect(audio_array, sample_rate, detected_language)
        
        # Generate explanation
        logger.info("Generating explanation")
        explanation = explainability_engine.explain(
            audio_array,
            sample_rate,
            detection_result,
            detected_language
        )
        
        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return AnalyzeResponse(
            classification=detection_result["classification"],
            confidence=detection_result["confidence"],
            language=detected_language,
            explanation=explanation,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/supported-languages")
async def get_supported_languages():
    return {
        "languages": settings.supported_languages
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
