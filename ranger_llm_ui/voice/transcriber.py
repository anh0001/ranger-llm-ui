"""
Speech-to-text using faster-whisper (local, offline).

Runs on Jetson GPU when CUDA + a matching CTranslate2 build are present, and
falls back to CPU int8 otherwise. The model is loaded lazily on first use so
importing this module is cheap and never fails just because the optional
``faster-whisper`` dependency is absent.

Configuration (environment variables):
    WHISPER_MODEL      faster-whisper model name (default: "small.en"; for the
                       best accuracy on a GPU box try "medium.en")
    WHISPER_DEVICE     "cuda", "cpu", or "auto" (default: "auto")
    WHISPER_COMPUTE    compute type override (default: device-dependent)
    WHISPER_LANGUAGE   transcription language hint (default: "en")
    WHISPER_BEAM_SIZE  beam search width (default: 5; 1 = greedy, less accurate)
    WHISPER_INITIAL_PROMPT  text prompt that biases decoding toward the robot
                       command vocabulary (default: built-in; set to "" to disable)
    WHISPER_HOTWORDS   space-separated terms to boost (default: built-in command
                       words; set to "" to disable)
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

# Bias Whisper toward the robot's small, fixed command vocabulary. Short
# utterances of confusable command words ("forward" vs "foreword", "right" vs
# "write") are exactly where an initial_prompt + hotwords help most.
_DEFAULT_INITIAL_PROMPT = (
    "Voice commands for a robot named Ranger. Commands include: emergency stop, "
    "stop, move forward, move backward, turn left, turn right, navigate to, "
    "battery status, system health, odometry, list nodes, list topics, camera "
    "image. Units include meters, centimeters, and degrees. Numbers matter."
)
_DEFAULT_HOTWORDS = (
    "Ranger emergency stop move forward move backward turn left turn right "
    "navigate battery system health odometry list nodes list topics camera image "
    "meters centimeters degrees"
)


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
        self.beam_size = max(1, int(os.getenv("WHISPER_BEAM_SIZE", "5") or 5))
        # Domain biasing. Empty env value explicitly disables (None).
        ip = os.getenv("WHISPER_INITIAL_PROMPT")
        self.initial_prompt = _DEFAULT_INITIAL_PROMPT if ip is None else (ip or None)
        hw = os.getenv("WHISPER_HOTWORDS")
        self.hotwords = _DEFAULT_HOTWORDS if hw is None else (hw or None)
        # VAD off by default: Silero VAD needs onnxruntime, which aborts on Tegra.
        self.vad_filter = os.getenv("WHISPER_VAD", "").lower() in {
            "1", "true", "yes", "on"
        }
        # Confidence signals from the most recent transcription, for callers
        # (e.g. the LLM second-pass corrector) to gate on. Populated by
        # transcribe(): {"avg_logprob", "no_speech_prob", "compression_ratio"}.
        self.last_stats: dict = {}

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

        # Deterministic, domain-biased decoding. beam_size>1 + initial_prompt +
        # hotwords markedly improve short fixed-vocabulary command accuracy;
        # temperature=0 avoids "creative" fallbacks that invent wrong commands.
        params = dict(
            language=self.language,
            vad_filter=self.vad_filter,
            beam_size=self.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
            hotwords=self.hotwords,
            without_timestamps=True,
        )
        self.last_stats = {}
        try:
            try:
                segments, _info = self._model.transcribe(audio_path, **params)
            except TypeError:
                # Older faster-whisper: drop kwargs it doesn't accept (hotwords,
                # without_timestamps) and retry with the portable subset.
                for k in ("hotwords", "without_timestamps"):
                    params.pop(k, None)
                segments, _info = self._model.transcribe(audio_path, **params)

            seg_list = list(segments)
            text = " ".join(s.text.strip() for s in seg_list).strip()
            if seg_list:
                logps = [s.avg_logprob for s in seg_list if s.avg_logprob is not None]
                nsps = [s.no_speech_prob for s in seg_list if s.no_speech_prob is not None]
                crs = [s.compression_ratio for s in seg_list if s.compression_ratio is not None]
                self.last_stats = {
                    "avg_logprob": (sum(logps) / len(logps)) if logps else None,
                    "no_speech_prob": (max(nsps) if nsps else None),
                    "compression_ratio": (max(crs) if crs else None),
                }
            logger.info(
                "Transcribed %d chars from %s (avg_logprob=%s, no_speech=%s)",
                len(text), audio_path,
                self.last_stats.get("avg_logprob"),
                self.last_stats.get("no_speech_prob"),
            )
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
