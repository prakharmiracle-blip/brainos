"""
BrainOS — Voice Ingester
Transcribes audio files using OpenAI Whisper (runs locally).
"""

from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.utils.logger import log


class VoiceIngester:
    _model = None  # lazy-loaded to avoid startup cost if unused

    def _get_model(self):
        if self._model is None:
            import whisper
            log.info(f"Loading Whisper model: {settings.whisper_model}")
            self._model = whisper.load_model(settings.whisper_model)
            log.info("Whisper model loaded.")
        return self._model

    def transcribe(self, audio_path: str | Path) -> str:
        """
        Transcribe an audio file and return the text.
        Supports mp3, mp4, wav, m4a, ogg, webm.
        """
        audio_path = str(audio_path)
        log.info(f"Transcribing audio: {audio_path}")

        model = self._get_model()
        result = model.transcribe(audio_path, fp16=False)
        text = result["text"].strip()

        log.info(f"Transcription complete. Length: {len(text)} chars.")
        return text


# Singleton
voice_ingester = VoiceIngester()
