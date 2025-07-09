# pipeline.py (Corrected for XTTS-v2)
# ... (imports) ...
import numpy as np
import logging
import json

from models import ModelManager
import utils
import config

logger = logging.getLogger(__name__)

class ProcessingPipeline:
    def __init__(self, model_manager: ModelManager):
        self.models = model_manager
        self.conversation_history = []
        
    async def process(self, audio_float32: np.ndarray) -> np.ndarray:
        try:
            # 1. ASR
            logger.info("1. Transcribing user audio...")
            user_text = self.models.transcribe(audio_float32)
            if not user_text or len(user_text.strip()) < 2:
                return np.array([], dtype=np.float32)
            logger.info(f"User said: '{user_text}'")

            # 2. Emotion Classification
            logger.info("2. Classifying emotion...")
            emotion = self.models.classify_emotion(audio_float32)
            logger.info(f"Detected emotion: {emotion}")

            # 3. LLM - Simpler logic now
            logger.info("3. Generating LLM response...")
            response_text = self.models.generate_response(user_text, emotion, self.conversation_history)
            if not response_text:
                 return np.array([], dtype=np.float32)

            logger.info(f"AI will say: '{response_text}'")
            self.conversation_history.append({"role": "user", "content": user_text})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            self.conversation_history = self.conversation_history[-10:]

            # 4. TTS
            logger.info("4. Synthesizing speech...")
            # The call is simpler, no style_hint needed
            synthesized_wav = await self.models.synthesize_speech(response_text)
            
            resampled_wav = utils.resample_audio(synthesized_wav, config.AUDIO_SAMPLE_RATE_TTS, config.AUDIO_SAMPLE_RATE_OUT)
            logger.info("✅ Pipeline processing complete.")
            return resampled_wav

        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}", exc_info=True)
            return np.array([], dtype=np.float32)

    def reset_history(self):
        self.conversation_history.clear()
        logger.info("Conversation history reset.")
