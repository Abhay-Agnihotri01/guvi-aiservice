import numpy as np
import logging
from typing import Dict

from app.models.wav2vec_detector import Wav2VecDetector
from app.models.acoustic_classifier import AcousticClassifier
from app.models.spectral_autoencoder import SpectralAutoencoder
from app.models.gemini_detector import GeminiDetector
from app.config import settings

logger = logging.getLogger(__name__)

class EnsembleDetector:
    """
    Ensemble-based AI voice detection using multiple models
    """
    
    def __init__(self):
        logger.info("Initializing ensemble detector")
        
        # Initialize models
        self.wav2vec_detector = Wav2VecDetector()
        self.acoustic_classifier = AcousticClassifier()
        self.spectral_autoencoder = SpectralAutoencoder()
        self.gemini_detector = GeminiDetector()
        
        # Model weights
        self.wav2vec_weight = settings.wav2vec_weight
        self.acoustic_weight = settings.acoustic_weight
        self.spectral_weight = settings.spectral_weight
        self.gemini_weight = settings.gemini_weight
        
        logger.info("Ensemble detector initialized")
    
    def models_loaded(self) -> bool:
        """Check if all models are loaded"""
        return (
            self.wav2vec_detector.is_loaded() and
            self.acoustic_classifier.is_loaded() and
            self.spectral_autoencoder.is_loaded()
        )
    
    def detect(self, audio: np.ndarray, sample_rate: int, language: str) -> Dict:
        """
        Run ensemble detection on audio
        """
        logger.info("Running ensemble detection")
        
        # Get predictions from each model
        wav2vec_score = self.wav2vec_detector.predict(audio, sample_rate)
        acoustic_score = self.acoustic_classifier.predict(audio, sample_rate, language)
        spectral_score = self.spectral_autoencoder.predict(audio, sample_rate)
        
        
        # Gemini prediction (only if enabled and weighted)
        gemini_score = 0.5
        if self.gemini_weight > 0 and self.gemini_detector.is_loaded():
            try:
                gemini_score = self.gemini_detector.predict(audio, sample_rate)
            except Exception as e:
                logger.error(f"Gemini detection failed: {e}")
                gemini_score = 0.5 # Neutral score on error
        
        # Calculate weighted ensemble score
        # Note: If Gemini is disabled (weight=0), it contributes 0 to the sum
        ensemble_score = (
            self.wav2vec_weight * wav2vec_score +
            self.acoustic_weight * acoustic_score +
            self.spectral_weight * spectral_score +
            self.gemini_weight * gemini_score
        )
        
        # Normalize score if needed (e.g. if weights don't sum to 1.0 due to dynamic disabling)
        total_weight = self.wav2vec_weight + self.acoustic_weight + self.spectral_weight + self.gemini_weight
        if self.gemini_weight > 0 and not self.gemini_detector.is_loaded():
             # If it was supposed to be used but failed loading
             total_weight -= self.gemini_weight
        
        if total_weight > 0:
            ensemble_score = ensemble_score / total_weight
        
        # Determine classification
        classification = "AI_GENERATED" if ensemble_score > settings.ai_threshold else "HUMAN"
        
        # Determine agreement level
        scores = [wav2vec_score, acoustic_score, spectral_score, gemini_score]
        agreement = self._calculate_agreement(scores)
        
        result = {
            "classification": classification,
            "confidence": float(ensemble_score),
            "model_scores": {
                "wav2vec2": float(wav2vec_score),
                "acoustic": float(acoustic_score),
                "spectral": float(spectral_score),
                "gemini": float(gemini_score)
            },
            "ensemble_agreement": agreement
        }
        
        logger.info(f"Detection result: {classification} ({ensemble_score:.3f})")
        
        return result
    
    def _calculate_agreement(self, scores: list) -> str:
        """Calculate model agreement level"""
        threshold = settings.ai_threshold
        ai_votes = sum(1 for score in scores if score > threshold)
        
        if ai_votes == 3 or ai_votes == 0:
            return "High"
        elif ai_votes == 2 or ai_votes == 1:
            return "Moderate"
        else:
            return "Low"
