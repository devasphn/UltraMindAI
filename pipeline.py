# pipeline.py
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
        """
        Runs the full S2S pipeline: ASR -> Emotion -> LLM -> TTS.
        
        :param audio_float32: Input audio from the user (16kHz).
        :return: Synthesized audio response (48kHz).
        """
        try:
            # 1. ASR: Transcribe audio to text
            logger.info("1. Transcribing user audio...")
            user_text = self.models.transcribe(audio_float32)
            if not user_text or len(user_text.strip()) < 2:
                logger.info("Transcription empty, skipping.")
                return np.array([], dtype=np.float32)
            logger.info(f"User said: '{user_text}'")

            # 2. Emotion Classification
            logger.info("2. Classifying emotion...")
            emotion = self.models.classify_emotion(audio_float32)
            logger.info(f"Detected emotion: {emotion}")

            # 3. LLM: Generate response
            # Note: Memory retrieval would happen here.
            logger.info("3. Generating LLM response...")
            llm_output_raw = self.models.generate_response(user_text, emotion, self.conversation_history)
            logger.info(f"LLM Raw Output: {llm_output_raw}")

            # 4. Parse LLM output and update history
            try:
                # Find the JSON part of the output
                json_start = llm_output_raw.find('{')
                json_end = llm_output_raw.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = llm_output_raw[json_start:json_end]
                    parsed_output = json.loads(json_str)
                    response_text = parsed_output['response']
                    style_hint = parsed_output.get('style_hint', 'A normal, conversational voice.')
                else:
                    # Fallback if JSON is not found
                    response_text = llm_output_raw
                    style_hint = 'A normal, conversational voice.'
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not parse LLM JSON output: {e}. Using raw output.")
                response_text = llm_output_raw
                style_hint = 'A normal, conversational voice.'
            
            if not response_text:
                 logger.info("LLM returned empty response, skipping TTS.")
                 return np.array([], dtype=np.float32)
            
            logger.info(f"AI will say: '{response_text}' with style: '{style_hint}'")
            self.conversation_history.append({"role": "user", "content": user_text})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            # Trim history to keep it from growing too large
            self.conversation_history = self.conversation_history[-10:]

            # 5. TTS: Synthesize response
            logger.info("4. Synthesizing speech...")
            synthesized_wav = await self.models.synthesize_speech(response_text, style_hint)
            
            # Resample TTS output (24kHz) to WebRTC output (48kHz)
            resampled_wav = utils.resample_audio(synthesized_wav, config.AUDIO_SAMPLE_RATE_TTS, config.AUDIO_SAMPLE_RATE_OUT)
            logger.info("✅ Pipeline processing complete.")
            return resampled_wav

        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}", exc_info=True)
            return np.array([], dtype=np.float32)

    def reset_history(self):
        self.conversation_history.clear()
        logger.info("Conversation history reset.")
