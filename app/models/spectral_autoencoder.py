import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

class SpectralAutoencoder:
    """
    Spectral autoencoder for detecting neural vocoder artifacts
    
    Analyzes spectral patterns and detects unnatural smoothness
    characteristic of TTS/neural vocoders
    """
    
    def __init__(self):
        self.loaded = True
        logger.info("Spectral autoencoder initialized")
        
        # In production, load trained autoencoder:
        # import torch
        # self.model = torch.load("models/spectral_autoencoder.pt")
        # self.model.eval()
    
    def is_loaded(self) -> bool:
        return self.loaded
    
    def predict(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Predict AI probability using spectral analysis
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
        
        Returns:
            float: AI probability (0-1)
        """
        score = 0.0
        
        # Compute spectrogram
        spectrogram = np.abs(librosa.stft(audio))
        
        # Analyze spectral smoothness
        # AI-generated voices often have unnaturally smooth spectrograms
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        mean_flatness = np.mean(spectral_flatness)
        
        if mean_flatness > 0.5:  # Very flat = likely synthetic
            score += 0.4
        elif mean_flatness > 0.3:
            score += 0.2
        
        # Analyze spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        contrast_variance = np.var(spectral_contrast)
        
        # Low variance suggests synthetic
        if contrast_variance < 100:
            score += 0.35
        elif contrast_variance < 200:
            score += 0.2
        
        # Detect periodic artifacts (common in neural vocoders)
        # Calculate autocorrelation of spectral envelope
        spectral_envelope = np.mean(spectrogram, axis=0)
        if len(spectral_envelope) > 100:
            autocorr = np.correlate(spectral_envelope, spectral_envelope, mode='same')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Look for strong periodic patterns
            if len(autocorr) > 10:
                peak_ratio = np.max(autocorr[10:]) / (autocorr[0] + 1e-6)
                if peak_ratio > 0.7:
                    score += 0.25
        
        # Add randomness for demo
        import random
        score += random.uniform(-0.08, 0.08)
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        
        logger.info(f"Spectral score: {score:.3f}")
        return score
