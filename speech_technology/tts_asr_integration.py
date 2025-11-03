#!/usr/bin/env python3
"""
Advanced TTS/ASR Integration System
Integrates Whisper ASR and Tortoise TTS for enterprise applications

Author: Lukas Kurniawan
Purpose: Production-ready speech technology integration
"""

import asyncio
import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import threading
from queue import Queue

# Speech processing imports
import whisper
import librosa
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, save_audio
import soundfile as sf
import webrtcvad

# Infrastructure imports
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
ASR_REQUESTS = Counter('asr_requests_total', 'Total ASR requests')
TTS_REQUESTS = Counter('tts_requests_total', 'Total TTS requests')
PROCESSING_TIME = Histogram('processing_duration_seconds', 'Processing time')
ACTIVE_SESSIONS = Gauge('active_sessions', 'Active processing sessions')

@dataclass
class AudioConfig:
    """Configuration for audio processing"""
    sample_rate: int = 16000
    chunk_size: int = 1024
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    min_speech_duration: float = 0.5
    max_silence_duration: float = 2.0

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    whisper_model: str = "base"
    tts_model_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True
    batch_size: int = 4

class WhisperASREngine:
    """Enterprise Whisper ASR Engine with optimization"""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load and optimize Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.config.whisper_model}")
            self.model = whisper.load_model(
                self.config.whisper_model,
                device=self.config.device
            )
            
            # Optimize for inference
            if self.config.use_fp16 and self.config.device == "cuda":
                self.model = self.model.half()
            
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_data: np.ndarray, language: Optional[str] = None) -> Dict:
        """Transcribe audio with confidence scoring"""
        start_time = datetime.now()
        
        try:
            ASR_REQUESTS.inc()
            
            # Prepare audio
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Transcription options
            options = {
                "language": language,
                "task": "transcribe",
                "fp16": self.config.use_fp16,
                "temperature": 0.0,  # Deterministic output
                "best_of": 5,  # Multiple candidates for better quality
                "beam_size": 5,
                "patience": 1.0
            }
            
            # Perform transcription
            result = self.model.transcribe(audio_data, **options)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(result)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            PROCESSING_TIME.observe(processing_time)
            
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "confidence": confidence,
                "processing_time": processing_time,
                "segments": result.get("segments", []),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"ASR transcription failed: {str(e)}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e),
                "success": False
            }
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate overall confidence score"""
        if "segments" not in result or not result["segments"]:
            return 0.5
        
        # Average confidence from segments
        confidences = []
        for segment in result["segments"]:
            if "avg_logprob" in segment:
                # Convert log probability to confidence (0-1)
                conf = np.exp(segment["avg_logprob"])
                confidences.append(conf)
        
        return np.mean(confidences) if confidences else 0.5

class TortoiseTTSEngine:
    """Enterprise Tortoise TTS Engine with voice cloning"""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.tts = None
        self.voice_cache = {}
        self.load_model()
    
    def load_model(self):
        """Load Tortoise TTS model"""
        try:
            logger.info("Loading Tortoise TTS model")
            self.tts = TextToSpeech(
                models_dir=self.config.tts_model_path,
                enable_redaction=False
            )
            logger.info("Tortoise TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {str(e)}")
            raise
    
    def synthesize_speech(
        self, 
        text: str, 
        voice: str = "random",
        preset: str = "fast",
        **kwargs
    ) -> Dict:
        """Synthesize speech with voice customization"""
        start_time = datetime.now()
        
        try:
            TTS_REQUESTS.inc()
            
            # Load voice samples if not cached
            if voice not in self.voice_cache and voice != "random":
                self._load_voice_samples(voice)
            
            # Generate speech
            voice_samples = self.voice_cache.get(voice, None)
            conditioning_latents = None
            
            if voice_samples:
                conditioning_latents = self.tts.get_conditioning_latents(
                    voice_samples
                )
            
            gen = self.tts.tts_with_preset(
                text,
                voice_samples=voice_samples,
                conditioning_latents=conditioning_latents,
                preset=preset,
                **kwargs
            )
            
            # Convert to numpy array
            audio_data = gen.squeeze().cpu().numpy()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            PROCESSING_TIME.observe(processing_time)
            
            return {
                "audio_data": audio_data,
                "sample_rate": 22050,  # Tortoise default
                "processing_time": processing_time,
                "voice": voice,
                "text_length": len(text),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {str(e)}")
            return {
                "audio_data": np.array([]),
                "error": str(e),
                "success": False
            }
    
    def _load_voice_samples(self, voice: str):
        """Load voice samples for cloning"""
        try:
            voice_path = Path(f"voices/{voice}")
            if voice_path.exists():
                samples = []
                for audio_file in voice_path.glob("*.wav"):
                    audio = load_audio(str(audio_file), 22050)
                    samples.append(audio)
                
                self.voice_cache[voice] = samples
                logger.info(f"Loaded {len(samples)} voice samples for {voice}")
            
        except Exception as e:
            logger.error(f"Failed to load voice samples for {voice}: {str(e)}")

class VoiceActivityDetector:
    """Voice Activity Detection for real-time processing"""
    
    def __init__(self, audio_config: AudioConfig):
        self.config = audio_config
        self.vad = webrtcvad.Vad(audio_config.vad_aggressiveness)
        
    def is_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> bool:
        """Detect if audio chunk contains speech"""
        try:
            # WebRTC VAD requires specific sample rates
            if sample_rate not in [8000, 16000, 32000, 48000]:
                # Resample to 16kHz
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                audio_data = librosa.resample(
                    audio_data.astype(float), 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
                audio_chunk = (audio_data * 32767).astype(np.int16).tobytes()
                sample_rate = 16000
            
            return self.vad.is_speech(audio_chunk, sample_rate)
            
        except Exception as e:
            logger.error(f"VAD error: {str(e)}")
            return False

class SpeechProcessingPipeline:
    """Complete speech processing pipeline"""
    
    def __init__(self, audio_config: AudioConfig, model_config: ModelConfig):
        self.audio_config = audio_config
        self.model_config = model_config
        
        # Initialize components
        self.asr_engine = WhisperASREngine(model_config)
        self.tts_engine = TortoiseTTSEngine(model_config)
        self.vad = VoiceActivityDetector(audio_config)
        
        # Processing queues
        self.asr_queue = Queue()
        self.tts_queue = Queue()
        
        # Redis for caching
        self.redis_client = redis.Redis(host='localhost', port=6379, db=1)
        
        # Start processing threads
        self._start_processing_threads()
        
        # Start metrics server
        start_http_server(8000)
        
    def _start_processing_threads(self):
        """Start background processing threads"""
        threading.Thread(target=self._asr_worker, daemon=True).start()
        threading.Thread(target=self._tts_worker, daemon=True).start()
        
    def _asr_worker(self):
        """Background ASR processing worker"""
        while True:
            try:
                task = self.asr_queue.get()
                if task is None:
                    break
                
                ACTIVE_SESSIONS.inc()
                result = self.asr_engine.transcribe_audio(
                    task['audio_data'],
                    task.get('language')
                )
                
                # Call callback if provided
                if task.get('callback'):
                    task['callback'](result)
                
                ACTIVE_SESSIONS.dec()
                self.asr_queue.task_done()
                
            except Exception as e:
                logger.error(f"ASR worker error: {str(e)}")
                ACTIVE_SESSIONS.dec()
    
    def _tts_worker(self):
        """Background TTS processing worker"""
        while True:
            try:
                task = self.tts_queue.get()
                if task is None:
                    break
                
                ACTIVE_SESSIONS.inc()
                result = self.tts_engine.synthesize_speech(
                    task['text'],
                    task.get('voice', 'random'),
                    task.get('preset', 'fast')
                )
                
                # Call callback if provided
                if task.get('callback'):
                    task['callback'](result)
                
                ACTIVE_SESSIONS.dec()
                self.tts_queue.task_done()
                
            except Exception as e:
                logger.error(f"TTS worker error: {str(e)}")
                ACTIVE_SESSIONS.dec()
    
    def process_audio_stream(self, audio_stream, callback: Callable = None):
        """Process real-time audio stream"""
        speech_buffer = []
        silence_duration = 0
        
        for audio_chunk in audio_stream:
            # Voice activity detection
            is_speech = self.vad.is_speech(audio_chunk)
            
            if is_speech:
                speech_buffer.append(audio_chunk)
                silence_duration = 0
            else:
                silence_duration += len(audio_chunk) / self.audio_config.sample_rate
                
                # Process accumulated speech if silence threshold reached
                if (silence_duration > self.audio_config.max_silence_duration and 
                    len(speech_buffer) > 0):
                    
                    # Convert to numpy array
                    audio_data = np.concatenate([
                        np.frombuffer(chunk, dtype=np.int16) 
                        for chunk in speech_buffer
                    ])
                    
                    # Queue for ASR processing
                    self.asr_queue.put({
                        'audio_data': audio_data,
                        'callback': callback
                    })
                    
                    speech_buffer = []
    
    def transcribe_file(self, file_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe audio file"""
        try:
            # Load audio file
            audio_data, sr = librosa.load(file_path, sr=self.audio_config.sample_rate)
            
            # Transcribe
            result = self.asr_engine.transcribe_audio(audio_data, language)
            
            return result
            
        except Exception as e:
            logger.error(f"File transcription failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def synthesize_to_file(self, text: str, output_path: str, voice: str = "random") -> bool:
        """Synthesize speech and save to file"""
        try:
            result = self.tts_engine.synthesize_speech(text, voice)
            
            if result['success']:
                sf.write(
                    output_path,
                    result['audio_data'],
                    result['sample_rate']
                )
                logger.info(f"Speech synthesized and saved to {output_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Speech synthesis to file failed: {str(e)}")
            return False
    
    def get_system_stats(self) -> Dict:
        """Get system performance statistics"""
        return {
            "active_sessions": ACTIVE_SESSIONS._value._value,
            "total_asr_requests": ASR_REQUESTS._value._value,
            "total_tts_requests": TTS_REQUESTS._value._value,
            "queue_sizes": {
                "asr": self.asr_queue.qsize(),
                "tts": self.tts_queue.qsize()
            },
            "model_config": {
                "whisper_model": self.model_config.whisper_model,
                "device": self.model_config.device,
                "fp16_enabled": self.model_config.use_fp16
            }
        }

# Example usage
if __name__ == "__main__":
    audio_config = AudioConfig(
        sample_rate=16000,
        vad_aggressiveness=2
    )
    
    model_config = ModelConfig(
        whisper_model="base",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    pipeline = SpeechProcessingPipeline(audio_config, model_config)
    
    # Example: Transcribe file
    result = pipeline.transcribe_file("example_audio.wav")
    print(f"Transcription: {result}")
    
    # Example: Synthesize speech
    success = pipeline.synthesize_to_file(
        "Hello, this is a test of the speech synthesis system.",
        "output_speech.wav",
        voice="random"
    )
    print(f"Synthesis success: {success}")
    
    # Print system stats
    stats = pipeline.get_system_stats()
    print(f"System stats: {json.dumps(stats, indent=2)}")