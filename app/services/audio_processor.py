import io
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles audio file processing, format conversion, and preprocessing
    """
    
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
    
    def process_audio(self, audio_bytes: bytes) -> tuple:
        """
        Process audio bytes to numpy array
        
        Args:
            audio_bytes: Raw audio file bytes
        
        Returns:
            tuple: (audio_array, sample_rate)
        """
        try:
            # Try to load directly with soundfile
            try:
                audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
            except:
                # If soundfile fails, try pydub for format conversion
                logger.info("Using pydub for audio conversion")
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                
                # Convert to wav
                wav_io = io.BytesIO()
                audio_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                
                audio_array, sample_rate = sf.read(wav_io)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample to target sample rate
            if sample_rate != self.target_sr:
                logger.info(f"Resampling from {sample_rate}Hz to {self.target_sr}Hz")
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sample_rate,
                    target_sr=self.target_sr
                )
                sample_rate = self.target_sr
            
            # Normalize audio
            audio_array = self.normalize(audio_array)
            
            logger.info(f"Audio processed: {len(audio_array)} samples at {sample_rate}Hz")
            
            return audio_array, sample_rate
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise ValueError(f"Failed to process audio: {str(e)}")
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    def extract_features(self, audio: np.ndarray, sr: int) -> dict:
        """
        Extract acoustic features from audio
        
        Returns:
            dict: Dictionary of audio features
        """
        features = {}
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Pitch
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_variance'] = np.var(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_variance'] = 0
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        zero_crossings = librosa.feature.zero_crossing_rate(audio)
        features['zero_crossing_rate'] = np.mean(zero_crossings)
        
        return features
