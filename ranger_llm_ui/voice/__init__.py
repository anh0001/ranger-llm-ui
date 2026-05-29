"""
Voice subsystem for Ranger LLM UI.

Provides local, offline speech-to-text (faster-whisper) and text-to-speech
(Piper) so the operator can talk to the robot and hear its replies without any
cloud API key. Both backends are lazy-loaded and degrade gracefully: if the
optional dependencies or models are missing, the rest of the UI keeps working
and the voice helpers report a clear status string instead of crashing.

Public API:
    get_transcriber() -> Transcriber   # speech-to-text (singleton)
    get_synthesizer() -> Synthesizer   # text-to-speech (singleton)
    voice_status() -> str              # human-readable availability summary
"""

from ranger_llm_ui.voice.transcriber import Transcriber, get_transcriber
from ranger_llm_ui.voice.synthesizer import Synthesizer, get_synthesizer


def voice_status() -> str:
    """Return a short summary of STT/TTS availability for the Settings tab."""
    stt = get_transcriber().status()
    tts = get_synthesizer().status()
    return f"STT (faster-whisper): {stt}\nTTS: {tts}"


__all__ = [
    "Transcriber",
    "Synthesizer",
    "get_transcriber",
    "get_synthesizer",
    "voice_status",
]
