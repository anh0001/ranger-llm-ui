"""
Speech-to-text using faster-whisper (local, offline).

Runs on Jetson GPU when CUDA + a matching CTranslate2 build are present, and
falls back to CPU int8 otherwise. The model is loaded lazily on first use so
importing this module is cheap and never fails just because the optional
``faster-whisper`` dependency is absent.

Configuration (environment variables):
    WHISPER_MODEL      faster-whisper model name (default: "small.en")
    WHISPER_DEVICE     "cuda", "cpu", or "auto" (default: "auto")
    WHISPER_COMPUTE    compute type override (default: device-dependent)
    WHISPER_LANGUAGE   transcription language hint (default: "en")
    WHISPER_VAD        "1"/"true" to enable Silero VAD filtering (default: off).
                       VAD pulls in onnxruntime, which hard-aborts on Jetson
                       Tegra CPUs, so it is OFF by default. Transcription itself
                       uses CTranslate2 and does not need onnxruntime.
"""

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "small.en"


class Transcriber:
    """Lazy faster-whisper wrapper. Thread-safe single model instance."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._error: Optional[str] = None
        self._loaded = False

        self.model_name = os.getenv("WHISPER_MODEL", _DEFAULT_MODEL)
        self.device = os.getenv("WHISPER_DEVICE", "auto").lower()
        self.compute_type = os.getenv("WHISPER_COMPUTE", "").strip() or None
        self.language = os.getenv("WHISPER_LANGUAGE", "en").strip() or None
        # VAD off by default: Silero VAD needs onnxruntime, which aborts on Tegra.
        self.vad_filter = os.getenv("WHISPER_VAD", "").lower() in {
            "1", "true", "yes", "on"
        }

    # -- internals ---------------------------------------------------------

    def _resolve_device(self) -> tuple[str, str]:
        """Pick (device, compute_type), honoring explicit overrides."""
        device = self.device
        if device == "auto":
            device = "cpu"
            try:
                import ctranslate2  # noqa: F401

                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
            except Exception:
                device = "cpu"

        if self.compute_type:
            compute = self.compute_type
        else:
            compute = "int8_float16" if device == "cuda" else "int8"
        return device, compute

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                self._error = (
                    "faster-whisper not installed "
                    "(pip install faster-whisper)"
                )
                self._loaded = True
                logger.warning(self._error)
                return

            device, compute = self._resolve_device()
            try:
                logger.info(
                    "Loading faster-whisper model '%s' (device=%s, compute=%s)",
                    self.model_name, device, compute,
                )
                self._model = WhisperModel(
                    self.model_name, device=device, compute_type=compute
                )
                logger.info("faster-whisper model ready")
            except Exception as e:
                # Common Jetson failure: CUDA/cuDNN mismatch. Retry on CPU once.
                if device == "cuda":
                    logger.warning(
                        "CUDA load failed (%s); falling back to CPU int8", e
                    )
                    try:
                        self._model = WhisperModel(
                            self.model_name, device="cpu", compute_type="int8"
                        )
                        logger.info("faster-whisper model ready on CPU")
                    except Exception as e2:
                        self._error = f"failed to load model: {e2}"
                        logger.error(self._error)
                else:
                    self._error = f"failed to load model: {e}"
                    logger.error(self._error)
            finally:
                self._loaded = True

    # -- public ------------------------------------------------------------

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def status(self) -> str:
        self._load()
        if self._model is not None:
            device, compute = self._resolve_device()
            return f"ready ({self.model_name}, {device}/{compute})"
        return f"unavailable — {self._error or 'unknown error'}"

    def transcribe(self, audio_path: Optional[str]) -> str:
        """
        Transcribe an audio file to text.

        Returns the transcript, or an empty string if there is no audio. On
        backend failure returns a short ``[voice: ...]`` marker so the caller
        can surface it without crashing the chat flow.
        """
        if not audio_path:
            return ""
        if not self.available:
            return f"[voice: STT {self._error or 'unavailable'}]"

        try:
            segments, _info = self._model.transcribe(
                audio_path,
                language=self.language,
                vad_filter=self.vad_filter,
                beam_size=1,
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            logger.info("Transcribed %d chars from %s", len(text), audio_path)
            return text
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return f"[voice: transcription error: {e}]"


_transcriber: Optional[Transcriber] = None
_transcriber_lock = threading.Lock()


def get_transcriber() -> Transcriber:
    """Return the process-wide Transcriber singleton."""
    global _transcriber
    if _transcriber is None:
        with _transcriber_lock:
            if _transcriber is None:
                _transcriber = Transcriber()
    return _transcriber
