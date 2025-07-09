# models.py (Corrected Version)

import torch
import whisperx
import librosa
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import yaml
import logging
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# --- Add StyleTTS2 to path ---
# This is the crucial correction.
# It allows us to import StyleTTS2 modules from the cloned directory.
style_tts_path = os.path.join(os.path.dirname(__file__), 'StyleTTS2')
if style_tts_path not in sys.path:
    sys.path.append(style_tts_path)
# --- End of correction ---

from phonemizer.backend import EspeakBackend
# This import now works because of the path change above
from style_tts2.tts import StyleTTS

import config
import utils

# Setup logger
logger = logging.getLogger(__name__)

class ModelManager:
    """A singleton class to load and manage all ML models."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            logger.info("Initializing ModelManager...")
            self.device = config.DEVICE
            self.compute_type = config.COMPUTE_TYPE
            self.executor = ThreadPoolExecutor(max_workers=3)
            
            self.asr_model = None
            self.emotion_model = None
            self.llm_model = None
            self.llm_tokenizer = None
            self.tts_model = None
            self.phonemizer = None
            
            self.initialized = True
            
    def load_models(self):
        """Loads all models into memory."""
        logger.info(f"Loading models on device: {self.device}")
        
        # 1. Load ASR Model (WhisperX) - No changes here
        logger.info(f"Loading ASR model: {config.ASR_MODEL}")
        self.asr_model = whisperx.load_model(
            config.ASR_MODEL,
            self.device,
            compute_type=self.compute_type,
            language="en"
        )
        logger.info("✅ ASR model loaded.")

        # 2. Load Emotion Recognition Model - No changes here
        logger.info(f"Loading Emotion model: {config.EMOTION_MODEL}")
        self.emotion_model = pipeline(
            "audio-classification",
            model=config.EMOTION_MODEL,
            device=self.device
        )
        logger.info("✅ Emotion model loaded.")

        # 3. Load LLM (Llama 3) - No changes here
        logger.info(f"Loading LLM: {config.LLM_MODEL}")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        logger.info("✅ LLM loaded.")

        # 4. Load TTS (StyleTTS 2) - This section is updated
        logger.info(f"Loading TTS model from local path: {config.TTS_MODEL_PATH}")
        if not os.path.exists(config.TTS_MODEL_PATH) or not os.path.exists(config.TTS_CONFIG_PATH):
            logger.error("StyleTTS2 model or config not found!")
            logger.error(f"Please make sure '{config.TTS_MODEL_PATH}' and '{config.TTS_CONFIG_PATH}' exist.")
            logger.error("Run the download commands from the setup instructions.")
            raise FileNotFoundError("StyleTTS2 model files not found.")

        with open(config.TTS_CONFIG_PATH, 'r') as f:
            tts_config = yaml.safe_load(f)

        self.tts_model = StyleTTS(
            text_aligner_path=os.path.join(config.TTS_MODEL_DIR, tts_config['data']['text_aligner_path']),
            style_encoder_path=os.path.join(config.TTS_MODEL_DIR, tts_config['model']['style_encoder_path']),
            model_path=config.TTS_MODEL_PATH
        )
        self.tts_model.to(self.device)
        self.phonemizer = EspeakBackend(language='en-us', with_stress=True)
        logger.info("✅ TTS model loaded.")
        
        logger.info("🎉 All models loaded successfully!")

    # The rest of the ModelManager class (transcribe, classify_emotion, generate_response, _blocking_synthesize, synthesize_speech)
    # remains exactly the same as before. No changes are needed there.
    def transcribe(self, audio_float32: np.ndarray) -> str:
        """Transcribes audio using WhisperX."""
        if self.asr_model is None:
            raise RuntimeError("ASR model not loaded.")
        result = self.asr_model.transcribe(audio_float32, batch_size=config.ASR_BATCH_SIZE)
        return " ".join([segment['text'] for segment in result['segments']]).strip()

    def classify_emotion(self, audio_float32: np.ndarray) -> str:
        """Classifies emotion from audio."""
        if self.emotion_model is None:
            raise RuntimeError("Emotion model not loaded.")
        
        if len(audio_float32) < 1000:
            return "neutral"
            
        result = self.emotion_model(
            {"raw": audio_float32, "sampling_rate": config.AUDIO_SAMPLE_RATE_EMO}
        )
        emotion = sorted(result, key=lambda x: x['score'], reverse=True)[0]['label']
        return emotion

    def generate_response(self, user_text: str, emotion: str, history: list) -> str:
        """Generates a text response using Llama 3."""
        if self.llm_model is None or self.llm_tokenizer is None:
            raise RuntimeError("LLM not loaded.")
        
        system_prompt = f"""You are Eva, an advanced, empathetic AI assistant. You are having a voice conversation.
Be concise, conversational, and human-like. The user sounds {emotion}.
Your task is to generate a JSON object with two keys: 'response' for what you will say, and 'style_hint' for how you will say it.
The style_hint should be a short phrase describing the desired tone, e.g., 'calm and reassuring', 'upbeat and friendly', 'thoughtful and curious'.
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        input_ids = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        terminators = [
            self.llm_tokenizer.eos_token_id,
            self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with torch.inference_mode():
            outputs = self.llm_model.generate(
                input_ids,
                max_new_tokens=150,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        response = outputs[0][input_ids.shape[-1]:]
        return self.llm_tokenizer.decode(response, skip_special_tokens=True)

    def _blocking_synthesize(self, text: str, style_hint: str) -> np.ndarray:
        """The actual TTS synthesis, run in a thread."""
        if self.tts_model is None or self.phonemizer is None:
            raise RuntimeError("TTS model not loaded.")
            
        phonemes = self.phonemizer.phonemize([text], strip=True)[0]
        
        with torch.inference_mode():
            output = self.tts_model.synthesize(
                phonemes,
                text_prompt=style_hint,
                style_speaker=None,
                lang='en-us',
                alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1.5
            )
            wav = output['wav'].cpu().numpy().flatten()
        return wav

    async def synthesize_speech(self, text: str, style_hint: str) -> np.ndarray:
        """Synthesizes speech from text using StyleTTS 2, non-blocking."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, self._blocking_synthesize, text, style_hint
        )
