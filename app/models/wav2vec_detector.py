import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import numpy as np
import logging
import os
from app.config import settings

logger = logging.getLogger(__name__)

class Wav2VecDetector:
    """
    Wav2Vec2-based deepfake detection model
    Uses the fine-tuned wav2vec2-base for voice activity/authenticity detection
    """
    
    def __init__(self):
        try:
            # Check for fine-tuned model in potential directories
            potential_paths = [
                os.path.join(settings.models_dir, "wav2vec_finetuned"),
                os.path.join(settings.models_dir, "my_deepfake_model")
            ]
            
            self.model_name = "facebook/wav2vec2-base" # Default fallback
            
            for path in potential_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                    logger.info(f"✅ Found fine-tuned model at: {path}")
                    self.model_name = path
                    break
            
            if self.model_name == "facebook/wav2vec2-base":
                logger.warning("❌ No custom fine-tuned model found. Using UNTRAINED base model.")

            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

            self.model.eval()
            self.loaded = True
            
            if "finetuned" in str(self.model_name):
                logger.info("✅ Wav2Vec detector successfully loaded with CUSTOM TRAINED MODEL")
            else:
                logger.info("⚠️ Wav2Vec detector initialized with BASE model")
                
        except Exception as e:
            logger.error(f"Error loading Wav2Vec2 model: {str(e)}")
            self.loaded = False
    
    def is_loaded(self) -> bool:
        return self.loaded
    
    def predict(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Predict AI probability score using Wav2Vec2
        """
        if not self.loaded:
            return 0.5
            
        try:
            # Preprocess
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=sample_rate, 
                return_tensors="pt", 
                padding=True
            )
            
            # Inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
                
            # Softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Assuming class 1 is AI-generated (this depends on the specific fine-tuning)
            ai_score = probs[0][1].item()
            
            logger.info(f"Wav2Vec2 neural score: {ai_score:.3f}")
            return ai_score
            
        except Exception as e:
            logger.error(f"Wav2Vec2 inference error: {str(e)}")
            return 0.5
