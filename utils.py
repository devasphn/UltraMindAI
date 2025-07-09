# utils.py
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resamples audio data to a target sample rate."""
    if orig_sr == target_sr:
        return audio_data
    try:
        return librosa.resample(y=audio_data, orig_sr=orig_sr, target_sr=target_sr)
    except Exception as e:
        logger.error(f"Error during resampling from {orig_sr} to {target_sr}: {e}")
        # Return original data as a fallback
        return audio_data

def float_to_int16(audio_float: np.ndarray) -> np.ndarray:
    """Converts float audio array to int16."""
    return (audio_float * 32767).astype(np.int16)

def int16_to_float32(audio_int16: np.ndarray) -> np.ndarray:
    """Converts int16 audio array to float32."""
    return audio_int16.astype(np.float32) / 32768.0
