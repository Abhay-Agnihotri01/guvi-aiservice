import google.generativeai as genai
import numpy as np
import logging
import io
import soundfile as sf
from app.config import settings

logger = logging.getLogger(__name__)

class GeminiDetector:
    """
    Multimodal audio analysis using Google Gemini 1.5 Pro
    """
    
    def __init__(self):
        self.api_key = settings.google_api_key
        self.model_name = "gemini-1.5-flash" 
        self.loaded = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self.loaded = True
                logger.info(f"Gemini detector initialized with {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {str(e)}")
        else:
            logger.warning("GOOGLE_API_KEY not found. Gemini detector will be disabled.")

    def is_loaded(self) -> bool:
        return self.loaded

    def predict(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """
        Analyze audio using Gemini and return AI probability score
        """
        if not self.loaded:
            logger.warning("Gemini not loaded, returning default score")
            return 0.5
            
        try:
            # Convert numpy array to WAV bytes for Gemini
            byte_io = io.BytesIO()
            sf.write(byte_io, audio_data, sample_rate, format='WAV')
            audio_bytes = byte_io.getvalue()
            
            prompt = """
            Analyze this audio file for authenticity. Is this a human speaker or an AI-generated/synthesized voice?
            Look for:
            1. Subtle robotic artifacts or unnatural phrasing.
            2. Spectral consistency that doesn't match human breath patterns.
            3. Prosody and emotional depth.
            
            Respond ONLY with a JSON object in this format:
            {"ai_probability": 0.XX, "reasoning": "short explanation"}
            """
            
            response = self.model.generate_content([
                prompt,
                {
                    "mime_type": "audio/wav",
                    "data": audio_bytes
                }
            ])
            
            # Extract JSON from response
            import json
            import re
            
            text = response.text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                score = float(result.get("ai_probability", 0.5))
                logger.info(f"Gemini score: {score:.3f} - Reasoning: {result.get('reasoning')}")
                return score
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Gemini analysis error: {str(e)}")
            return 0.5
