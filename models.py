# models.py (Corrected for XTTS-v2)
import torch
import whisperx
import librosa
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import logging
from concurrent.futures import ThreadPoolExecutor
import os
# The TTS import is now much simpler and more reliable
from TTS.api import TTS

import config
import utils

logger = logging.getLogger(__name__)

class ModelManager:
    # ... (no changes to __new__ or __init__) ...
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
            self.executor = ThreadPoolExecutor(max_workers=2)
            self.asr_model = None
            self.emotion_model = None
            self.llm_model = None
            self.llm_tokenizer = None
            # Change attribute name for clarity
            self.xtts_model = None
            self.initialized = True

    def load_models(self):
        """Loads all models into memory."""
        logger.info(f"Loading models on device: {self.device}")
        
        # ASR, Emotion, and LLM loading remain unchanged...
        logger.info(f"Loading ASR model: {config.ASR_MODEL}")
        self.asr_model = whisperx.load_model(config.ASR_MODEL, self.device, compute_type=self.compute_type, language="en")
        logger.info("✅ ASR model loaded.")

        logger.info(f"Loading Emotion model: {config.EMOTION_MODEL}")
        self.emotion_model = pipeline("audio-classification", model=config.EMOTION_MODEL, device=self.device)
        logger.info("✅ Emotion model loaded.")

        logger.info(f"Loading LLM: {config.LLM_MODEL}")
        self.llm_model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        logger.info("✅ LLM loaded.")

        # --- 4. Load TTS (XTTS-v2) - THE NEW, WORKING METHOD ---
        logger.info(f"Loading TTS model: {config.TTS_MODEL}")
        if not os.path.exists(config.TTS_SPEAKER_WAV):
            raise FileNotFoundError(f"Reference speaker WAV file not found at '{config.TTS_SPEAKER_WAV}'. Please create it.")
        
        self.xtts_model = TTS(model_name=config.TTS_MODEL, progress_bar=True).to(self.device)
        logger.info("✅ TTS model loaded.")
        
        logger.info("🎉 All models loaded successfully!")

    # ... (transcribe, classify_emotion, generate_response remain unchanged) ...
    def transcribe(self, audio_float32: np.ndarray) -> str:
        if self.asr_model is None: raise RuntimeError("ASR model not loaded.")
        result = self.asr_model.transcribe(audio_float32, batch_size=config.ASR_BATCH_SIZE)
        return " ".join([segment['text'] for segment in result['segments']]).strip()

    def classify_emotion(self, audio_float32: np.ndarray) -> str:
        if self.emotion_model is None: raise RuntimeError("Emotion model not loaded.")
        if len(audio_float32) < 1000: return "neutral"
        result = self.emotion_model({"raw": audio_float32, "sampling_rate": config.AUDIO_SAMPLE_RATE_EMO})
        return sorted(result, key=lambda x: x['score'], reverse=True)[0]['label']

    def generate_response(self, user_text: str, emotion: str, history: list) -> str:
        if self.llm_model is None or self.llm_tokenizer is None: raise RuntimeError("LLM not loaded.")
        # We can simplify the LLM prompt now, as we don't need a text style_hint
        system_prompt = f"""You are Eva, an advanced, empathetic AI assistant in a voice call. The user sounds {emotion}. Be concise, conversational, and human-like."""
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_text}]
        input_ids = self.llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        terminators = [self.llm_tokenizer.eos_token_id, self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        with torch.inference_mode():
            outputs = self.llm_model.generate(input_ids, max_new_tokens=150, eos_token_id=terminators, do_sample=True, temperature=0.7, top_p=0.9)
        response = outputs[0][input_ids.shape[-1]:]
        return self.llm_tokenizer.decode(response, skip_special_tokens=True).strip()

    # --- Synthesize Speech - THE NEW, WORKING METHOD ---
    def _blocking_synthesize(self, text: str) -> np.ndarray:
        """The actual TTS synthesis, run in a thread."""
        if self.xtts_model is None:
            raise RuntimeError("TTS model not loaded.")
            
        # The `tts` method directly returns a numpy array, which is perfect.
        wav = self.xtts_model.tts(
            text=text,
            speaker_wav=config.TTS_SPEAKER_WAV,
            language="en"
        )
        return np.array(wav)

    async def synthesize_speech(self, text: str) -> np.ndarray:
        """Synthesizes speech from text using XTTS-v2, non-blocking."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, self._blocking_synthesize, text
        )
