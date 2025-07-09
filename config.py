# config.py
import torch

# --- General Settings ---
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
LOG_LEVEL = "INFO"

# --- WebRTC & Audio Settings ---
STUN_SERVER = "stun:stun.l.google.com:19302"
AUDIO_SAMPLE_RATE_IN = 48000  # Input from WebRTC
AUDIO_SAMPLE_RATE_VAD = 16000 # Required by Silero VAD
AUDIO_SAMPLE_RATE_ASR = 16000 # Required by Whisper
AUDIO_SAMPLE_RATE_EMO = 16000 # Required by Emotion Model
AUDIO_SAMPLE_RATE_TTS = 24000 # Output from StyleTTS2
AUDIO_SAMPLE_RATE_OUT = 48000 # Required by WebRTC

# --- VAD Settings ---
VAD_MIN_SPEECH_MS = 250
VAD_SPEECH_PAD_MS = 100
VAD_THRESHOLD = 0.5

# --- Model Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "default"

# ASR (WhisperX)
ASR_MODEL = "large-v3"
ASR_BATCH_SIZE = 16

# Emotion Recognition
EMOTION_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

# LLM (Llama 3)
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# TTS (StyleTTS 2)
TTS_MODEL = "yl4579/StyleTTS2-LibriTTS"
TTS_CONFIG = "yl4579/StyleTTS2-LibriTTS/config.yml"

# --- Memory Settings ---
MEMORY_DB_PATH = "./chroma_db"
MEMORY_COLLECTION = "conversation_history"
