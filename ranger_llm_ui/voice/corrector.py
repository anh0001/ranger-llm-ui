"""
LLM second-pass correction for voice transcripts.

faster-whisper gives a decent but imperfect transcript of a short spoken robot
command. This module runs an OPTIONAL second pass with the app's existing LLM
(the same provider the chat agent uses, e.g. claude_proxy) to normalize obvious
ASR errors against the robot's known command vocabulary BEFORE the text reaches
the agent — e.g. "move foreword to meters" -> "move forward two meters".

Design (see docs / Codex review):
* It is a *normalizer*, not the agent. Output is plain corrected text.
* Hard safety bypass: anything that reads as a stop / emergency-stop is returned
  verbatim — the LLM is never given a chance to turn a stop into motion.
* Confidence gate: clean, high-confidence transcripts skip the LLM entirely
  (no latency cost). Only low-confidence / unparseable transcripts are sent.
* Fail-safe: any LLM or parsing error returns the raw transcript unchanged.

Configuration (environment variables):
    VOICE_LLM_CORRECTION    "1"/"true" to enable (default: on)
    VOICE_CORRECTION_MODEL  model for the correction call (default: the chat
                            model; set e.g. "haiku-4.5" for lower latency)
    VOICE_CORRECTION_LOGPROB_GATE   skip LLM above this avg_logprob (default -0.55)
    VOICE_CORRECTION_NOSPEECH_GATE  skip LLM below this no_speech_prob (default 0.35)
"""

import json
import logging
import os
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# A transcript that already reads as a stop command must never be rewritten.
_STOP_RE = re.compile(
    r"\b(emergency\s+stop|e[-\s]?stop|stop|halt|abort|freeze)\b", re.IGNORECASE
)
# Whisper error markers like "[voice: ...]" should pass through untouched.
_MARKER_RE = re.compile(r"^\s*\[voice:")

_SYSTEM_PROMPT = (
    "You correct short speech-to-text transcripts of spoken commands for a robot "
    "named Ranger. Map the transcript to the closest valid command.\n\n"
    "Allowed command families:\n"
    "- emergency stop\n- stop\n"
    "- move forward <number> meters|centimeters\n"
    "- move backward <number> meters|centimeters\n"
    "- turn left <number> degrees\n- turn right <number> degrees\n"
    "- navigate to <location>\n"
    "- battery status\n- system health\n- odometry\n"
    "- list nodes\n- list topics\n- camera image\n\n"
    "Rules:\n"
    "- Output JSON only, no prose.\n"
    "- Fix obvious ASR homophones: foreword->forward, write/wright->right, "
    "to/too->two, for->four, ate->eight, metered/meter->meters, degree->degrees.\n"
    "- Preserve numbers, units, directions and destinations; never invent a "
    "missing distance, angle, direction or destination.\n"
    "- Never turn a stop into a movement command.\n"
    "- If the transcript does not resemble any allowed command, set "
    'action="keep" and return it unchanged.\n'
    "- If it is gibberish or empty, set action=\"reject\".\n\n"
    "JSON schema: {\"action\":\"keep|correct|reject\","
    "\"corrected_text\":string,\"confidence\":number,\"reason\":string}"
)


class TranscriptCorrector:
    """Lazy, thread-safe LLM corrector that reuses the app's LLM provider."""

    def __init__(self) -> None:
        self._llm = None
        self._lock = threading.Lock()
        self._loaded = False
        self._error: Optional[str] = None

        self.enabled = os.getenv("VOICE_LLM_CORRECTION", "1").lower() in {
            "1", "true", "yes", "on"
        }
        self.logprob_gate = float(
            os.getenv("VOICE_CORRECTION_LOGPROB_GATE", "-0.55") or -0.55
        )
        self.nospeech_gate = float(
            os.getenv("VOICE_CORRECTION_NOSPEECH_GATE", "0.35") or 0.35
        )

    def _build_llm(self, provider, model: Optional[str]):
        if self._loaded:
            return self._llm
        with self._lock:
            if self._loaded:
                return self._llm
            try:
                from ranger_llm_ui.agent_interface import create_llm
                corr_model = os.getenv("VOICE_CORRECTION_MODEL") or model
                # Deterministic, non-streaming, short single-shot call.
                self._llm = create_llm(
                    provider=provider,
                    model_name=corr_model,
                    temperature=0.0,
                    streaming=False,
                )
                logger.info("Voice corrector LLM ready (model=%s)", corr_model)
            except Exception as e:
                self._error = f"corrector LLM unavailable: {e}"
                logger.warning(self._error)
                self._llm = None
            finally:
                self._loaded = True
        return self._llm

    @staticmethod
    def _high_confidence(stats: dict, logprob_gate: float, nospeech_gate: float) -> bool:
        if not stats:
            return False
        lp = stats.get("avg_logprob")
        ns = stats.get("no_speech_prob")
        if lp is None:
            return False
        if lp <= logprob_gate:
            return False
        if ns is not None and ns >= nospeech_gate:
            return False
        return True

    def correct(self, raw_text: str, stats: Optional[dict],
                provider, model: Optional[str]) -> str:
        """Return a corrected transcript, or ``raw_text`` unchanged on any
        skip/failure. Always safe to call; never raises."""
        text = (raw_text or "").strip()
        if not self.enabled or not text:
            return raw_text
        # Pass through Whisper error markers untouched.
        if _MARKER_RE.match(text):
            return raw_text
        # Hard safety bypass: stop commands are never rewritten by the LLM.
        if _STOP_RE.search(text):
            return raw_text
        # High-confidence, likely-clean transcript -> skip the LLM (no latency).
        if self._high_confidence(stats or {}, self.logprob_gate, self.nospeech_gate):
            return raw_text

        llm = self._build_llm(provider, model)
        if llm is None:
            return raw_text

        try:
            from langchain_core.messages import HumanMessage
            payload = json.dumps({
                "raw_transcript": text,
                "avg_logprob": (stats or {}).get("avg_logprob"),
                "no_speech_prob": (stats or {}).get("no_speech_prob"),
            })
            # Put the whole instruction in the USER turn. Some backends (notably
            # claude_proxy, which wraps Claude Code) inject their own assistant
            # persona and ignore a system message, replying like a code helper;
            # a single user turn that ends with a hard JSON-only constraint is
            # interpreted correctly across providers.
            user = (
                _SYSTEM_PROMPT
                + "\n\nTranscript to correct (JSON input):\n" + payload
                + "\n\nRespond with ONLY the JSON object, nothing else."
            )
            resp = llm.invoke([HumanMessage(content=user)])
            out = self._extract_json(getattr(resp, "content", resp))
            if not out:
                return raw_text

            action = str(out.get("action", "keep")).lower()
            corrected = (out.get("corrected_text") or "").strip()
            conf = out.get("confidence")
            try:
                conf = float(conf) if conf is not None else 1.0
            except (TypeError, ValueError):
                conf = 1.0

            if action == "correct" and corrected:
                # Re-check safety: a correction must not introduce a stop flip
                # nor (defensively) turn a non-stop into something odd at low
                # confidence. Accept only reasonably confident corrections.
                if _STOP_RE.search(corrected) and not _STOP_RE.search(text):
                    return raw_text
                if conf < 0.6:
                    return raw_text
                if corrected.lower() != text.lower():
                    logger.info("Voice corrected %r -> %r (conf=%.2f, %s)",
                                text, corrected, conf, out.get("reason", ""))
                return corrected
            # action == "keep" / "reject" / anything else -> leave raw text;
            # the agent itself will ask for clarification if needed.
            return raw_text
        except Exception as e:
            logger.warning("Voice correction failed (%s); using raw transcript", e)
            return raw_text

    @staticmethod
    def _extract_json(content) -> Optional[dict]:
        """Pull the first JSON object out of an LLM message (handles code
        fences and list-of-blocks content shapes)."""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        if not isinstance(content, str):
            content = str(content)
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


_corrector: Optional[TranscriptCorrector] = None
_corrector_lock = threading.Lock()


def get_corrector() -> TranscriptCorrector:
    """Return the process-wide TranscriptCorrector singleton."""
    global _corrector
    if _corrector is None:
        with _corrector_lock:
            if _corrector is None:
                _corrector = TranscriptCorrector()
    return _corrector
