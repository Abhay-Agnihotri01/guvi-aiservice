import numpy as np
import logging
import os
from typing import Optional
from app.utils.acoustic_language_detector import AcousticLanguageDetector

logger = logging.getLogger(__name__)

class LanguageDetector:
    """
    Detects language from audio using multiple methods:
    1. User-provided hint (highest priority)
    2. SpeechBrain model (if available)
    3. Acoustic feature analysis (fallback)
    """
    
    def __init__(self):
        self.supported_languages = ["tamil", "english", "hindi", "malayalam", "telugu"]
        self.acoustic_detector = AcousticLanguageDetector()
        self._speechbrain_model = None
        self._speechbrain_failed = False
    
    def detect(self, audio: np.ndarray, sample_rate: int, language_hint: str = None) -> str:
        """
        Detect language from audio.
        Prioritizes language hint, then uses SpeechBrain for audio-based detection with robust error handling.
        """
        # Priority 1: Use language hint if provided
        if language_hint:
            if language_hint.lower() in self.supported_languages:
                logger.info(f"✅ Using provided language hint: {language_hint}")
                return language_hint.lower()
        
        # Priority 2: Try SpeechBrain (only if not previously failed)
        if not self._speechbrain_failed:
            speechbrain_result = self._try_speechbrain(audio, sample_rate)
            if speechbrain_result:
                return speechbrain_result
        
        # Priority 3: Use acoustic feature-based detection (always works)
        logger.info("🎵 Using acoustic-based language detection...")
        detected_lang = self.acoustic_detector.detect(audio, sample_rate)
        logger.info(f"✅ Acoustic detector result: {detected_lang}")
        return detected_lang
    
    def _try_speechbrain(self, audio: np.ndarray, sample_rate: int) -> Optional[str]:
        """Try SpeechBrain language detection, return None if fails"""
        try:
            # Try to load model if not loaded yet
            if self._speechbrain_model is None:
                logger.info("📥 Loading SpeechBrain language detection model...")
                
                from speechbrain.pretrained import EncoderClassifier
                import torch
                
                # Use absolute path for cache directory
                cache_dir = os.path.abspath("speechbrain_cache")
                os.makedirs(cache_dir, exist_ok=True)
                
                self._speechbrain_model = EncoderClassifier.from_hparams(
                    source="speechbrain/lang-id-voxlingua107-ecapa", 
                    savedir=cache_dir,
                    run_opts={"device": "cpu"},
                    use_auth_token=False
                )
                logger.info("✅ SpeechBrain model loaded successfully")
            
            # Run prediction
            import torch
            signal = torch.tensor(audio).float()
            if signal.dim() == 1:
                signal = signal.unsqueeze(0)
            
            prediction = self._speechbrain_model.classify_batch(signal)
            _, best_probs, best_lang_id = prediction
            detected_code = best_lang_id[0].lower()
            
            # Map language codes to full names
            code_map = {
                "hi": "hindi", 
                "ta": "tamil", 
                "te": "telugu", 
                "ml": "malayalam", 
                "en": "english"
            }
            
            # Check if detected code matches our supported languages
            for code, name in code_map.items():
                if code in detected_code:
                    logger.info(f"✅ SpeechBrain detected language: {name} (code: {detected_code})")
                    return name
            
            logger.warning(f"Detected unsupported language code: {detected_code}")
            return None
                    
        except OSError as e:
            logger.warning(f"❌ SpeechBrain OS Error (Windows symlink issue): {str(e)[:100]}...")
            logger.info("💡 Falling back to acoustic-based detection")
            self._speechbrain_failed = True  # Don't try again
            return None
            
        except ImportError as e:
            logger.warning(f"❌ SpeechBrain not installed: {e}")
            self._speechbrain_failed = True
            return None
            
        except Exception as e:
            logger.warning(f"❌ SpeechBrain detection failed: {e}")
            self._speechbrain_failed = True
            return None
