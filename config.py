# config.py (Corrected Version)
import torch
import os

# --- General Settings ---
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
LOG_LEVEL = "INFO"

# --- WebRTC & Audio Settings ---
STUN_SERVER = "stun:stun.l.google.com:19302"
AUDIO_SAMPLE_RATE_IN = 48000
AUDIO_SAMPLE_RATE_VAD = 16000
AUDIO_SAMPLE_RATE_ASR = 16000
AUDIO_SAMPLE_RATE_EMO = 16000
AUDIO_SAMPLE_RATE_TTS = 24000
AUDIO_SAMPLE_RATE_OUT = 48000

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

# TTS (StyleTTS 2) - Paths are now local
# We will create these directories and files in the setup commands
TTS_MODEL_DIR = os.path.join("StyleTTS2", "Models", "LibriTTS")
TTS_MODEL_PATH = os.path.join(TTS_MODEL_DIR, "epochs_2nd_00020.pth")
TTS_CONFIG_PATH = os.path.join(TTS_MODEL_DIR, "config.yml")

# --- Memory Settings ---
MEMORY_DB_PATH = "./chroma_db"
MEMORY_COLLECTION = "conversation_history"
