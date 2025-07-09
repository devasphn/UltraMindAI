# config.py (Corrected for XTTS-v2)
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
# XTTS outputs at 24kHz, which we will upsample
AUDIO_SAMPLE_RATE_TTS = 24000
AUDIO_SAMPLE_RATE_OUT = 48000

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

# TTS (XTTS-v2) - New, simpler, and more robust configuration
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
# IMPORTANT: Create this file. Record 5-10 seconds of a clear, neutral voice.
TTS_SPEAKER_WAV = "reference_voice.wav"
