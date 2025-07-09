# main.py
import asyncio
import json
import logging
import numpy as np
import fractions
import collections
import time
import av
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay

import config
import utils
from models import ModelManager
from pipeline import ProcessingPipeline

# --- Basic Setup ---
try:
    import uvloop
    uvloop.install()
    logging.info("🚀 Using uvloop for asyncio event loop.")
except ImportError:
    logging.info("⚠️ uvloop not found, using default asyncio event loop.")

logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('aiortc').setLevel(logging.WARNING)
logging.getLogger('aioice').setLevel(logging.WARNING)

# --- Global Variables ---
pcs = set()
model_manager = ModelManager()
relay = MediaRelay()

# --- VAD ---
class SileroVAD:
    def __init__(self):
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        (self.get_speech_timestamps, _, _, _, _) = self.utils

    def __call__(self, audio_chunk: torch.Tensor):
        return len(self.get_speech_timestamps(audio_chunk, self.model, sampling_rate=config.AUDIO_SAMPLE_RATE_VAD)) > 0

vad_model = SileroVAD()

# --- Audio Processing Classes ---
class AudioBuffer:
    def __init__(self):
        self.buffer = collections.deque()
        self.last_speech_time = time.time()
        self.silence_duration = 0.8  # seconds of silence to trigger processing

    def add_frame(self, frame: av.AudioFrame):
        self.buffer.append(frame)

    def should_process(self) -> bool:
        # Simple VAD logic: process after a pause in speech.
        # More advanced: process chunks as they come, reset on silence.
        # This implementation processes the whole buffer after a period of silence.
        
        # We need to get the latest audio chunk and check for speech
        if not self.buffer:
            return False

        latest_frame = self.buffer[-1]
        audio_float32 = utils.int16_to_float32(latest_frame.to_ndarray().flatten())
        resampled_for_vad = utils.resample_audio(audio_float32, latest_frame.sample_rate, config.AUDIO_SAMPLE_RATE_VAD)

        if vad_model(torch.from_numpy(resampled_for_vad)):
            self.last_speech_time = time.time()
            return False
        
        is_silence = (time.time() - self.last_speech_time) > self.silence_duration
        has_content = len(self.buffer) > 5 # at least some frames
        
        return is_silence and has_content

    def get_audio_array(self) -> np.ndarray:
        # Concatenate all frames in buffer
        full_audio_list = []
        target_sr = config.AUDIO_SAMPLE_RATE_ASR
        
        for frame in list(self.buffer):
            audio_float32 = utils.int16_to_float32(frame.to_ndarray().flatten())
            resampled = utils.resample_audio(audio_float32, frame.sample_rate, target_sr)
            full_audio_list.append(resampled)
        
        self.buffer.clear()
        return np.concatenate(full_audio_list) if full_audio_list else np.array([], dtype=np.float32)

class ResponseAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue()
        self._current_chunk = None
        self._chunk_pos = 0
        self._timestamp = 0
        self.sample_rate = config.AUDIO_SAMPLE_RATE_OUT

    async def recv(self) -> av.AudioFrame:
        frame_samples = 960  # 20ms at 48kHz
        frame = np.zeros(frame_samples, dtype=np.int16)

        if self._current_chunk is None or self._chunk_pos >= len(self._current_chunk):
            try:
                # Wait for a new chunk of audio data
                self._current_chunk = await asyncio.wait_for(self._queue.get(), timeout=0.02)
                self._chunk_pos = 0
            except asyncio.TimeoutError:
                # If no data, send silence
                pass

        if self._current_chunk is not None:
            samples_to_write = min(frame_samples, len(self._current_chunk) - self._chunk_pos)
            frame[:samples_to_write] = self._current_chunk[self._chunk_pos : self._chunk_pos + samples_to_write]
            self._chunk_pos += samples_to_write

        audio_frame = av.AudioFrame.from_ndarray(frame.reshape(1, -1), format="s16", layout="mono")
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = self.sample_rate
        self._timestamp += frame_samples
        return audio_frame

    async def queue_audio(self, audio_float32: np.ndarray):
        if audio_float32.size > 0:
            audio_int16 = utils.float_to_int16(audio_float32)
            await self._queue.put(audio_int16)

class AudioProcessor:
    def __init__(self, output_track: ResponseAudioTrack, pipeline: ProcessingPipeline):
        self.track = None
        self.buffer = AudioBuffer()
        self.output_track = output_track
        self.pipeline = pipeline
        self.task = None
        self.is_speaking = False

    def add_track(self, track: MediaStreamTrack):
        self.track = relay.subscribe(track)

    async def start(self):
        self.task = asyncio.create_task(self._run())

    async def stop(self):
        if self.task:
            self.task.cancel()
            self.task = None

    async def _run(self):
        try:
            while True:
                if self.is_speaking:
                    # While AI is speaking, discard incoming audio to prevent echo
                    await asyncio.sleep(0.02)
                    continue

                try:
                    frame = await asyncio.wait_for(self.track.recv(), timeout=0.1)
                    self.buffer.add_frame(frame)
                except asyncio.TimeoutError:
                    # Check for end-of-speech on timeout
                    pass
                
                if self.buffer.should_process():
                    audio_to_process = self.buffer.get_audio_array()
                    if audio_to_process.size > config.AUDIO_SAMPLE_RATE_ASR * 0.2: # min length
                        logger.info(f"🧠 Processing {len(audio_to_process) / config.AUDIO_SAMPLE_RATE_ASR:.2f}s of audio...")
                        # Do not await this, run it in the background
                        asyncio.create_task(self.process_and_play(audio_to_process))

        except asyncio.CancelledError:
            logger.info("Audio processor task cancelled.")
        except Exception as e:
            logger.error(f"Audio processor error: {e}", exc_info=True)
        finally:
            logger.info("Audio processor stopped.")

    async def process_and_play(self, audio_array: np.ndarray):
        self.is_speaking = True # Set state to speaking immediately
        try:
            response_audio = await self.pipeline.process(audio_array)
            if response_audio.size > 0:
                logger.info("🤖 AI is speaking...")
                await self.output_track.queue_audio(response_audio)
                playback_duration = response_audio.size / config.AUDIO_SAMPLE_RATE_OUT
                await asyncio.sleep(playback_duration + 0.2) # Wait for playback to finish + buffer
        except Exception as e:
            logger.error(f"Error during process_and_play: {e}", exc_info=True)
        finally:
            self.is_speaking = False
            logger.info("✅ AI finished speaking, now listening.")


# --- WebRTC and WebSocket Handling ---
async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    
    pc = RTCPeerConnection(RTCConfiguration([RTCIceServer(urls=config.STUN_SERVER)]))
    pcs.add(pc)
    processor = None
    pipeline_instance = ProcessingPipeline(model_manager)

    @pc.on("track")
    def on_track(track):
        nonlocal processor
        logger.info(f"🎧 Track {track.kind} received")
        if track.kind == "audio":
            output_audio_track = ResponseAudioTrack()
            pc.addTrack(output_audio_track)
            processor = AudioProcessor(output_audio_track, pipeline_instance)
            processor.add_track(track)
            asyncio.create_task(processor.start())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"ICE Connection State is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            if pc in pcs: pcs.remove(pc)
            if processor: await processor.stop()
            await pc.close()

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data["type"] == "offer":
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type=data["type"]))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket connection closed.")
        if processor: await processor.stop()
        if pc in pcs: pcs.remove(pc)
        if pc.connectionState != "closed": await pc.close()
    return ws

async def index_handler(request):
    return web.FileResponse('./static/index.html')

# --- Main Application Logic ---
async def on_startup(app):
    logger.info("Server starting up...")
    model_manager.load_models()
    logger.info("Model loading complete. Server is ready.")

async def on_shutdown(app):
    logger.info("Shutting down server...")
    for pc_conn in list(pcs): await pc_conn.close()
    pcs.clear()
    logger.info("Shutdown complete.")

async def main():
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_static('/static', 'static')

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, config.SERVER_HOST, config.SERVER_PORT)
    await site.start()
    
    print(f"✅ Server started successfully on http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    print("🚀 Your speech-to-speech AI agent is live!")
    print("   Press Ctrl+C to stop the server.")
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down by user request...")
