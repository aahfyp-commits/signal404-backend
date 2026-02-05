import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
import tempfile
import shutil

from app.config import TARGET_SR, TARGET_DURATION, N_MFCC, N_FFT, HOP_LENGTH

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.target_sr = TARGET_SR
        self.target_duration = TARGET_DURATION
        self.n_mfcc = N_MFCC
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
    
    def process_file(self, file_path: Path) -> np.ndarray:
        """
        Complete pipeline: load → preprocess → extract features
        
        Returns:
            Feature array of shape (time_steps, n_mfcc * 3)
        """
        try:
            # Step 1: Convert to WAV if needed and load
            wav_path = self._convert_to_wav(file_path)
            
            # Step 2: Preprocess (match training exactly)
            audio = self._preprocess(wav_path)
            
            # Step 3: Extract features
            features = self._extract_features(audio)
            
            # Cleanup temp file
            if wav_path != file_path:
                wav_path.unlink(missing_ok=True)
            
            return features
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise
    
    def _convert_to_wav(self, file_path: Path) -> Path:
        """Convert audio to WAV format if needed"""
        if file_path.suffix.lower() == '.wav':
            return file_path
        
        # Create temp WAV file
        temp_wav = Path(tempfile.gettempdir()) / f"{file_path.stem}_converted.wav"
        
        try:
            audio, sr = librosa.load(str(file_path), sr=self.target_sr)
            sf.write(str(temp_wav), audio, self.target_sr)
            return temp_wav
        except Exception as e:
            raise ValueError(f"Cannot convert {file_path.suffix} to WAV: {e}")
    
    def _preprocess(self, audio_path: Path) -> np.ndarray:
        """
        Preprocess audio EXACTLY like training:
        1. Load at target SR
        2. Trim silence
        3. Pad/truncate to 3 seconds
        4. Normalize
        """
        # Load
        y, sr = librosa.load(str(audio_path), sr=self.target_sr)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Fixed length
        target_length = int(self.target_sr * self.target_duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        return y
    
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC + delta + delta-delta features
        
        Returns:
            Array of shape (time_steps, n_mfcc * 3)
        """
        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.target_sr, 
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # Deltas
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack and transpose to (time_steps, features)
        combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        return combined.T
    
    def get_audio_info(self, file_path: Path) -> dict:
        """Get audio file metadata"""
        try:
            info = sf.info(str(file_path))
            return {
                "duration": round(info.duration, 2),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "format": info.format
            }
        except Exception as e:
            return {"error": str(e)}

# Global instance
processor = AudioProcessor()