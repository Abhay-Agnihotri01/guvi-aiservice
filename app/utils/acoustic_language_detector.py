import numpy as np
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

class AcousticLanguageDetector:
    """
    Acoustic-based language detector for Indian languages.
    Uses MFCC, pitch, formants, and rhythm features for classification.
    """
    
    def __init__(self):
        # Language-specific MFCC patterns (learned from typical speech samples)
        # These are approximate but useful for discrimination
        self.language_signatures = {
            "tamil": {
                "mfcc_mean": [15, -25, 10, -5, 3, -2, 1, 0, -1, 0, 1, 0, -1],
                "pitch_mean": 210,
                "pitch_std": 45,
                "zero_crossing_rate": 0.15,
                "spectral_rolloff": 3200,
            },
            "hindi": {
                "mfcc_mean": [12, -22, 8, -3, 2, -1, 0, 1, -1, 0, 0, 1, 0],
                "pitch_mean": 185,
                "pitch_std": 40,
                "zero_crossing_rate": 0.12,
                "spectral_rolloff": 3000,
            },
            "telugu": {
                "mfcc_mean": [14, -24, 9, -4, 2.5, -1.5, 0.5, 0, -0.5, 0, 0.5, 0, -0.5],
                "pitch_mean": 200,
                "pitch_std": 42,
                "zero_crossing_rate": 0.14,
                "spectral_rolloff": 3100,
            },
            "malayalam": {
                "mfcc_mean": [13, -23, 9, -4, 2.8, -1.8, 0.8, 0.2, -0.8, 0.1, 0.6, 0.1, -0.6],
                "pitch_mean": 195,
                "pitch_std": 43,
                "zero_crossing_rate": 0.16,
                "spectral_rolloff": 3150,
            },
            "english": {
                "mfcc_mean": [10, -20, 6, -2, 1, 0, 0, 0.5, -0.5, 0, 0, 0.5, -0.5],
                "pitch_mean": 160,
                "pitch_std": 35,
                "zero_crossing_rate": 0.10,
                "spectral_rolloff": 2800,
            }
        }
    
    def detect(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Detect language based on acoustic features.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            str: Detected language
        """
        try:
            import librosa
            
            # Process only first 30 seconds to avoid timeout on long files
            # 30 seconds is sufficient for accurate language detection
            max_samples = 30 * sample_rate
            audio_sample = audio[:max_samples] if len(audio) > max_samples else audio
            
            logger.info(f"Processing {len(audio_sample)/sample_rate:.1f} seconds of audio for language detection")
            
            # Extract multiple features
            features = self._extract_all_features(audio_sample, sample_rate)
            
            logger.info(f"Extracted features: Pitch={features['pitch_mean']:.1f}Hz, "
                       f"ZCR={features['zero_crossing_rate']:.3f}, "
                       f"Rolloff={features['spectral_rolloff']:.0f}Hz")
            
            # Score each language
            scores = {}
            for lang, signature in self.language_signatures.items():
                score = self._compute_similarity_score(features, signature)
                scores[lang] = score
            
            # Get best match
            best_lang = max(scores, key=scores.get)
            best_score = scores[best_lang]
            
            logger.info(f"Language scores: {scores}, Best match: {best_lang} (score: {best_score:.2f})")
            
            # Return best match if score is reasonable
            if best_score > 0.3:  # Lowered threshold for better fallback
                return best_lang
            else:
                logger.warning(f"Very low confidence, defaulting to Hindi")
                return "hindi"  # Changed default to Hindi instead of English
                
        except Exception as e:
            logger.error(f"Acoustic language detection failed: {e}", exc_info=True)
            return "hindi"  # Default to Hindi since it's common in the dataset
    
    def _extract_all_features(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """Extract comprehensive acoustic features"""
        import librosa
        
        features = {}
        
        # 1. MFCC features (most discriminative)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
        
        # 2. Pitch/F0 with robust extraction
        pitch_mean, pitch_std = self._extract_pitch_robust(audio, sample_rate)
        features['pitch_mean'] = pitch_mean
        features['pitch_std'] = pitch_std
        
        # 3. Zero Crossing Rate (rhythm/consonant measure)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zero_crossing_rate'] = float(np.mean(zcr))
        
        # 4. Spectral rolloff (frequency distribution)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        features['spectral_rolloff'] = float(np.mean(rolloff))
        
        return features
    
    def _extract_pitch_robust(self, audio: np.ndarray, sample_rate: int) -> Tuple[float, float]:
        """Extract pitch with multiple fallback methods"""
        import librosa
        
        try:
            # Method 1: pyin (most accurate but can fail)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=80,   # Minimum human voice
                fmax=400,  # Maximum for normal speech
                sr=sample_rate,
                frame_length=2048
            )
            
            # Filter out unvoiced and NaN values
            valid_f0 = f0[(voiced_flag) & (~np.isnan(f0))]
            
            if len(valid_f0) > 10:  # Need sufficient voiced frames
                pitch_mean = float(np.median(valid_f0))
                pitch_std = float(np.std(valid_f0))
                
                # Sanity check: typical human voice is 80-400Hz
                if 80 <= pitch_mean <= 400:
                    return pitch_mean, pitch_std
        except Exception as e:
            logger.debug(f"pyin failed: {e}")
        
        # Method 2: Autocorrelation-based pitch detection (fallback)
        try:
            # Simple autocorrelation method
            pitch_estimates = []
            frame_length = 2048
            hop_length = 512
            
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i+frame_length]
                
                # Autocorrelation
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find first peak after zero lag
                # Look for peaks between 80-400 Hz (sample_rate/400 to sample_rate/80 samples)
                min_lag = int(sample_rate / 400)
                max_lag = int(sample_rate / 80)
                
                if max_lag < len(autocorr):
                    search_region = autocorr[min_lag:max_lag]
                    if len(search_region) > 0:
                        peak_lag = min_lag + np.argmax(search_region)
                        if autocorr[peak_lag] > 0.3 * autocorr[0]:  # Threshold
                            f0_estimate = sample_rate / peak_lag
                            if 80 <= f0_estimate <= 400:
                                pitch_estimates.append(f0_estimate)
            
            if len(pitch_estimates) > 5:
                return float(np.median(pitch_estimates)), float(np.std(pitch_estimates))
        except Exception as e:
            logger.debug(f"Autocorrelation pitch failed: {e}")
        
        # Method 3: Default based on spectral centroid
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            avg_centroid = np.mean(spectral_centroids)
            # Rough estimate: pitch correlates with centroid
            estimated_pitch = np.clip(avg_centroid / 10, 120, 250)
            return float(estimated_pitch), 30.0
        except:
            pass
        
        # Ultimate fallback
        logger.warning("All pitch extraction methods failed, using default")
        return 180.0, 35.0  # Neutral pitch for Indian languages
    
    def _compute_similarity_score(self, features: Dict, signature: Dict) -> float:
        """Compute similarity score between extracted features and language signature"""
        score = 0.0
        
        # 1. MFCC similarity (most important - 50% weight)
        try:
            mfcc_dist = np.linalg.norm(
                np.array(features['mfcc_mean']) - np.array(signature['mfcc_mean'])
            )
            # Normalize: lower distance = higher score
            mfcc_score = max(0, 1 - (mfcc_dist / 50))  # Normalize by typical distance
            score += mfcc_score * 0.5
        except:
            pass
        
        # 2. Pitch similarity (25% weight)
        try:
            pitch_diff = abs(features['pitch_mean'] - signature['pitch_mean'])
            pitch_score = max(0, 1 - (pitch_diff / 100))  # 100Hz tolerance
            score += pitch_score * 0.25
        except:
            pass
        
        # 3. Zero crossing rate (15% weight)
        try:
            zcr_diff = abs(features['zero_crossing_rate'] - signature['zero_crossing_rate'])
            zcr_score = max(0, 1 - (zcr_diff / 0.1))
            score += zcr_score * 0.15
        except:
            pass
        
        # 4. Spectral rolloff (10% weight)
        try:
            rolloff_diff = abs(features['spectral_rolloff'] - signature['spectral_rolloff'])
            rolloff_score = max(0, 1 - (rolloff_diff / 1000))
            score += rolloff_score * 0.10
        except:
            pass
        
        return score
