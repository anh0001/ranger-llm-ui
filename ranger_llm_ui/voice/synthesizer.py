"""
Text-to-speech for Ranger LLM UI (local, offline).

Three backends, auto-selected best-first:

* **Kokoro** — 82M neural TTS on the PyTorch backend. Most natural English
  voice and, crucially, avoids onnxruntime entirely. Preferred.
* **Piper** — neural ONNX voices, natural. Used if Kokoro is unavailable AND
  onnxruntime is healthy on this platform.
* **espeak-ng** — classic formant synth, robotic but rock-solid. Pure C, no
  onnxruntime, works everywhere including NVIDIA Tegra/Jetson. Last resort.

Why this layering: the stock PyPI ``onnxruntime`` aarch64 wheel **hard-aborts
on import** on Jetson Tegra CPUs ("Unknown CPU vendor" -> C++ assertion). That
abort cannot be caught — it kills the whole process — so Piper (which needs
onnxruntime) is gated behind a *subprocess* health probe, while Kokoro uses
PyTorch and sidesteps the problem. Kokoro's torch needs ``libcudss.so.0``,
which we ctypes-preload from the installed ``nvidia`` wheel so no
LD_LIBRARY_PATH is required. The generated WAV is streamed to the browser by
Gradio, so no server-side audio device is needed.

Configuration (environment variables):
    TTS_BACKEND        "auto" (default), "kokoro", "piper", or "espeak"
    KOKORO_VOICE       Kokoro voice (default: "af_heart"; e.g. af_bella, am_michael)
    KOKORO_LANG        Kokoro lang code (default: "a" = American English)
    PIPER_VOICE        Piper voice name (default: "en_US-lessac-medium")
    PIPER_VOICE_PATH   explicit path to a .onnx voice file (skips download)
    PIPER_VOICE_DIR    cache dir for downloaded voices
                       (default: ~/.ranger_llm_ui/voices)
    PIPER_DOWNLOAD     "1"/"true" to allow auto-download (default: enabled)
    ESPEAK_VOICE       espeak-ng voice (default: "en-us")
    ESPEAK_SPEED       words per minute (default: 165)
"""

import ctypes
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

logger = logging.getLogger(__name__)

_DEFAULT_VOICE = "en_US-lessac-medium"
_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Strip UI/markdown noise that should not be spoken aloud.
_TOOL_LINE = re.compile(r"^Used tool:.*$", re.MULTILINE)
_MD_CHARS = re.compile(r"[*_`#>]+")
_USAGE_FOOTER = re.compile(r"\n---\nTokens:.*$", re.DOTALL)

# Cached onnxruntime health probe result (None = not yet probed).
_ort_ok: Optional[bool] = None
_ort_lock = threading.Lock()


def _onnxruntime_healthy() -> bool:
    """
    Return True if ``import onnxruntime`` succeeds without aborting.

    Runs in a subprocess because a bad wheel aborts the interpreter (SIGABRT),
    which a normal try/except cannot trap. Result is cached.
    """
    global _ort_ok
    if _ort_ok is not None:
        return _ort_ok
    with _ort_lock:
        if _ort_ok is not None:
            return _ort_ok
        try:
            proc = subprocess.run(
                [sys.executable, "-c", "import onnxruntime"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            _ort_ok = proc.returncode == 0
            if not _ort_ok:
                logger.warning(
                    "onnxruntime import unhealthy (rc=%s) — Piper disabled, "
                    "using espeak-ng fallback", proc.returncode,
                )
        except Exception as e:
            logger.warning("onnxruntime probe failed: %s", e)
            _ort_ok = False
    return _ort_ok


_cuda_preloaded = False
_cuda_lock = threading.Lock()


def _preload_cuda_libs() -> None:
    """
    ctypes-preload NVIDIA shared libs (libcudss etc.) bundled in pip wheels.

    Jetson torch wheels link ``libcudss.so.0`` which lives under
    ``site-packages/nvidia/.../lib`` and is not on the default loader path.
    Preloading with RTLD_GLOBAL makes the symbols available to ``import torch``
    without requiring LD_LIBRARY_PATH in the launch environment.
    """
    global _cuda_preloaded
    if _cuda_preloaded:
        return
    with _cuda_lock:
        if _cuda_preloaded:
            return
        # Preload ONLY the libs torch fails to find on its own (libcudss).
        # Preloading the whole nvidia/ tree with RTLD_GLOBAL causes symbol
        # clashes that later break transformers' lazy imports, so keep it
        # surgical — torch resolves the rest (cublas/cudnn) via /usr/local/cuda.
        for sp in sys.path:
            for lib in glob.glob(
                os.path.join(sp, "nvidia", "**", "lib", "libcudss.so*"),
                recursive=True,
            ):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        _cuda_preloaded = True


def _voice_url_parts(voice: str) -> tuple[str, str]:
    """Map 'en_US-lessac-medium' -> HF relative path for .onnx + .onnx.json."""
    lang_region, speaker, quality = voice.split("-", 2)
    lang = lang_region.split("_")[0]
    rel = f"{lang}/{lang_region}/{speaker}/{quality}/{voice}"
    return f"{rel}.onnx", f"{rel}.onnx.json"


class Synthesizer:
    """Lazy TTS wrapper. Selects Piper or espeak-ng. Thread-safe."""

    def __init__(self) -> None:
        self._voice = None              # loaded PiperVoice (if backend==piper)
        self._kokoro = None             # KPipeline (if backend==kokoro)
        self._kokoro_sr = 24000         # Kokoro output sample rate
        self._backend: Optional[str] = None  # "kokoro" | "piper" | "espeak" | None
        self._lock = threading.Lock()
        self._error: Optional[str] = None
        self._loaded = False

        self.requested = os.getenv("TTS_BACKEND", "auto").lower()
        self.kokoro_voice = os.getenv("KOKORO_VOICE", "af_heart")
        self.kokoro_lang = os.getenv("KOKORO_LANG", "a")
        self.voice_name = os.getenv("PIPER_VOICE", _DEFAULT_VOICE)
        self.voice_path = os.getenv("PIPER_VOICE_PATH", "").strip() or None
        self.cache_dir = Path(
            os.getenv("PIPER_VOICE_DIR")
            or (Path.home() / ".ranger_llm_ui" / "voices")
        )
        self.allow_download = os.getenv("PIPER_DOWNLOAD", "1").lower() in {
            "1", "true", "yes", "on"
        }
        self.espeak_voice = os.getenv("ESPEAK_VOICE", "en-us")
        self.espeak_speed = os.getenv("ESPEAK_SPEED", "165")
        self._espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")

    # -- Piper voice acquisition ------------------------------------------

    def _download(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        logger.info("Downloading Piper asset: %s", url)
        with urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dest)

    def _ensure_voice_file(self) -> Optional[Path]:
        if self.voice_path:
            p = Path(self.voice_path)
            if p.exists():
                return p
            self._error = f"PIPER_VOICE_PATH not found: {p}"
            return None

        onnx = self.cache_dir / f"{self.voice_name}.onnx"
        cfg = self.cache_dir / f"{self.voice_name}.onnx.json"
        if onnx.exists() and cfg.exists():
            return onnx
        if not self.allow_download:
            self._error = (
                f"voice '{self.voice_name}' not cached and download disabled"
            )
            return None
        try:
            onnx_rel, cfg_rel = _voice_url_parts(self.voice_name)
            if not onnx.exists():
                self._download(f"{_HF_BASE}/{onnx_rel}", onnx)
            if not cfg.exists():
                self._download(f"{_HF_BASE}/{cfg_rel}", cfg)
            return onnx
        except Exception as e:
            self._error = f"voice download failed: {e}"
            logger.error(self._error)
            return None

    def _try_load_kokoro(self) -> bool:
        """Attempt to set up the Kokoro backend. Returns True on success."""
        try:
            _preload_cuda_libs()
            from kokoro import KPipeline
        except Exception as e:
            self._error = f"kokoro unavailable: {e}"
            return False
        try:
            # Cap torch CPU threads so synthesis doesn't starve the agent /
            # web loop on the Jetson's limited cores (avoids chat timeouts).
            try:
                import torch
                torch.set_num_threads(int(os.getenv("KOKORO_THREADS", "4")))
            except Exception:
                pass
            self._kokoro = KPipeline(lang_code=self.kokoro_lang)
            self._backend = "kokoro"
            logger.info("Kokoro TTS ready: voice=%s lang=%s",
                        self.kokoro_voice, self.kokoro_lang)
            return True
        except Exception as e:
            self._error = f"failed to init Kokoro: {e}"
            logger.error(self._error)
            return False

    def _try_load_piper(self) -> bool:
        """Attempt to set up the Piper backend. Returns True on success."""
        if not _onnxruntime_healthy():
            self._error = "onnxruntime unhealthy on this platform"
            return False
        try:
            from piper import PiperVoice
        except ImportError:
            self._error = "piper-tts not installed"
            return False
        model_path = self._ensure_voice_file()
        if model_path is None:
            return False
        try:
            self._voice = PiperVoice.load(str(model_path))
            self._backend = "piper"
            logger.info("Piper voice ready: %s", self.voice_name)
            return True
        except Exception as e:
            self._error = f"failed to load Piper voice: {e}"
            logger.error(self._error)
            return False

    # -- backend selection -------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                want = self.requested
                if want == "espeak":
                    self._backend = "espeak" if self._espeak_bin else None
                    if not self._backend:
                        self._error = "espeak-ng not installed"
                elif want == "kokoro":
                    self._try_load_kokoro()
                elif want == "piper":
                    self._try_load_piper()
                else:  # auto: Kokoro (most natural) -> Piper -> espeak-ng
                    if not self._try_load_kokoro() and not self._try_load_piper():
                        if self._espeak_bin:
                            self._backend = "espeak"
                            logger.info("TTS using espeak-ng fallback")
                        else:
                            self._error = (
                                (self._error or "")
                                + "; espeak-ng also unavailable"
                            )
            finally:
                self._loaded = True

    # -- synthesis ---------------------------------------------------------

    def _write_wav_piper(self, text: str, wav_file: "wave.Wave_write") -> None:
        """Synthesize into an open wave file across Piper API versions."""
        voice = self._voice
        if hasattr(voice, "synthesize_wav"):
            voice.synthesize_wav(text, wav_file)
            return
        chunks = list(voice.synthesize(text))
        if chunks and hasattr(chunks[0], "audio_int16_bytes"):
            first = chunks[0]
            wav_file.setnchannels(getattr(first, "sample_channels", 1))
            wav_file.setsampwidth(getattr(first, "sample_width", 2))
            wav_file.setframerate(first.sample_rate)
            for c in chunks:
                wav_file.writeframes(c.audio_int16_bytes)
            return
        voice.synthesize(text, wav_file)

    def _synth_kokoro(self, text: str, out_path: str) -> None:
        import numpy as np
        import soundfile as sf

        chunks = [
            audio for _gs, _ps, audio in self._kokoro(text, voice=self.kokoro_voice)
        ]
        if not chunks:
            raise RuntimeError("Kokoro produced no audio")
        audio = np.concatenate(chunks)
        sf.write(out_path, audio, self._kokoro_sr)

    def _synth_espeak(self, text: str, out_path: str) -> None:
        subprocess.run(
            [
                self._espeak_bin,
                "-v", self.espeak_voice,
                "-s", str(self.espeak_speed),
                "-w", out_path,
                text,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=True,
        )

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove tool/markdown/usage noise so only real prose is spoken."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = _USAGE_FOOTER.sub("", text or "")
        text = _TOOL_LINE.sub("", text)
        text = _MD_CHARS.sub("", text)
        return " ".join(text.split()).strip()

    @property
    def available(self) -> bool:
        self._load()
        return self._backend is not None

    def status(self) -> str:
        self._load()
        if self._backend == "kokoro":
            return f"ready (Kokoro: {self.kokoro_voice})"
        if self._backend == "piper":
            return f"ready (Piper: {self.voice_name})"
        if self._backend == "espeak":
            return f"ready (espeak-ng: {self.espeak_voice})"
        return f"unavailable — {self._error or 'unknown error'}"

    def synthesize(self, text: str) -> Optional[str]:
        """
        Synthesize ``text`` to a temp WAV file and return its path.

        Returns None when there is nothing to say or no backend is available,
        so the caller can simply skip audio output.
        """
        text = self.clean_text(text)
        if not text:
            return None
        if not self.available:
            logger.warning("TTS skipped: %s", self._error)
            return None

        try:
            fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="ranger_tts_")
            os.close(fd)
            if self._backend == "kokoro":
                self._synth_kokoro(text, out_path)
            elif self._backend == "piper":
                with wave.open(out_path, "wb") as wf:
                    self._write_wav_piper(text, wf)
            else:
                self._synth_espeak(text, out_path)
            return out_path
        except Exception as e:
            logger.error("Synthesis failed (%s): %s", self._backend, e)
            return None


_synthesizer: Optional[Synthesizer] = None
_synthesizer_lock = threading.Lock()


def get_synthesizer() -> Synthesizer:
    """Return the process-wide Synthesizer singleton."""
    global _synthesizer
    if _synthesizer is None:
        with _synthesizer_lock:
            if _synthesizer is None:
                _synthesizer = Synthesizer()
    return _synthesizer
