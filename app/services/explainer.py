import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """
    Generate human-readable explanations for detection results
    """
    
    def __init__(self):
        # Baseline values for comparison (from human voice data)
        self.human_baselines = {
            "pitch_variance": 500.0,  # Typical human pitch variance
            "spectral_smoothness": 0.3,  # Spectral variance threshold
        }
    
    def explain(self, audio: np.ndarray, sample_rate: int, 
                detection_result: Dict, language: str) -> Dict:
        """
        Generate explanation for detection result
        """
        reasoning = []
        
        # 1. Base technical reasoning (Heuristics + Model scores)
        from app.services.audio_processor import AudioProcessor
        processor = AudioProcessor()
        features = processor.extract_features(audio, sample_rate)
        
        # Analyze pitch variance
        pitch_anomaly = False
        if features['pitch_variance'] < self.human_baselines['pitch_variance'] * 0.6:
            reasoning.append(
                f"Technical Analysis: Pitch variance {int((1 - features['pitch_variance']/self.human_baselines['pitch_variance']) * 100)}% "
                "lower than human baseline, suggesting synthetic monotonous pattern."
            )
            pitch_anomaly = True
            
        model_scores = detection_result['model_scores']
        if model_scores['wav2vec2'] > 0.8:
            reasoning.append(f"Neural Check: Wav2Vec2 detected high probability of vocoder artifacts ({model_scores['wav2vec2']:.2f}).")

        # 2. Rich reasoning via Gemini (if available)
        from app.models.gemini_detector import GeminiDetector
        gemini = GeminiDetector()
        
        if gemini.is_loaded():
            try:
                # Use Gemini to generate a human-like summary of all data
                prompt = f"""
                As an AI Voice Authenticity Expert, provide a concise explanation for this detection result.
                - Classification: {detection_result['classification']}
                - Confidence: {detection_result['confidence']:.2f}
                - Language: {language}
                - Technical Model Scores: {model_scores}
                
                Provide a single paragraph explanation (max 3 sentences) that is easy for a non-technical user to understand.
                Focus on why the models might have flagged this as {detection_result['classification']}.
                """
                
                import google.generativeai as genai
                genai.configure(api_key=gemini.api_key)
                model = genai.GenerativeModel("gemini-1.5-pro")
                response = model.generate_content(prompt)
                
                gemini_reasoning = response.text.strip()
                if gemini_reasoning:
                    reasoning.insert(0, f"Expert Insight: {gemini_reasoning}")
            except Exception as e:
                logger.error(f"Gemini reasoning error: {str(e)}")
        
        # If no specific issues found and Gemini failed
        if len(reasoning) == 0:
            reasoning.append("Audio characteristics match human voice patterns across all verified models.")
        
        explanation = {
            "reasoning": reasoning,
            "model_scores": model_scores,
            "pitch_anomaly": pitch_anomaly,
            "spectral_artifacts": model_scores['spectral'] > 0.8,
            "ensemble_agreement": detection_result['ensemble_agreement']
        }
        
        logger.info(f"Generated {len(reasoning)} explanation points")
        
        return explanation
