import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import logging

from app.config import MODEL_PATH, METADATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDetector:
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to load model only once"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.metadata = {}
        self._load_model()
        self._initialized = True
    
    def _load_model(self):
        """Load TensorFlow model and metadata"""
        try:
            logger.info(f"Loading model from: {MODEL_PATH}")
            self.model = keras.models.load_model(MODEL_PATH)
            
            logger.info(f"Loading metadata from: {METADATA_PATH}")
            with open(METADATA_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Override config with metadata if available
            self.n_mfcc = self.metadata.get('n_mfcc', 40)
            self.n_fft = self.metadata.get('n_fft', 2048)
            self.hop_length = self.metadata.get('hop_length', 512)
            self.max_pad_len = self.metadata.get('max_pad_len', 94)
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"   Input shape: ({self.max_pad_len}, {self.n_mfcc * 3})")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def predict(self, features: np.ndarray, threshold: float = 0.3) -> dict:
        """
        Run prediction on preprocessed features
        
        Args:
            features: Shape (time_steps, n_features) - MFCC + deltas
            threshold: Decision threshold (0.3 works better for your model)
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Pad sequence
            padded = self._pad_sequence(features, self.max_pad_len)
            input_data = np.expand_dims(padded, axis=0)  # Add batch dim
            
            # Predict
            probability = float(self.model.predict(input_data, verbose=0)[0][0])
            
            # Interpret (0=bonafide, 1=spoofed in training)
            is_spoofed = probability > threshold
            confidence = probability * 100 if is_spoofed else (1 - probability) * 100
            
            return {
                "is_fake": bool(is_spoofed),
                "is_real": not bool(is_spoofed),
                "prediction": "FAKE" if is_spoofed else "REAL",
                "confidence": round(confidence, 2),
                "fake_probability": round(probability, 4),
                "real_probability": round(1 - probability, 4),
                "threshold_used": threshold
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _pad_sequence(self, sequence: np.ndarray, maxlen: int) -> np.ndarray:
        """Pad or truncate sequence to fixed length"""
        n_features = sequence.shape[1]
        padded = np.zeros((maxlen, n_features))
        
        if len(sequence) > maxlen:
            padded = sequence[:maxlen]
        else:
            padded[:len(sequence)] = sequence
            
        return padded
    
    def health_check(self) -> dict:
        """Check if model is loaded and ready"""
        return {
            "model_loaded": self.model is not None,
            "metadata_loaded": bool(self.metadata),
            "input_shape": [None, self.max_pad_len, self.n_mfcc * 3] if self.model else None
        }

# Global instance
detector = DeepfakeDetector()