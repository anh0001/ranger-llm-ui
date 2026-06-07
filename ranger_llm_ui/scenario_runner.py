"""
Scenario runner + safety supervisor for the Ranger LLM UI.

This module executes a :class:`~ranger_llm_ui.scenarios.Scenario` one step at a
time against the agent (the equivalent of looping
``claude -p "$prompt" --continue`` over a prompts file), while enforcing a
two-layer safety net so the robot stops — or recovers — when a step goes wrong:

1. **Deterministic tripwire** (:func:`looks_like_error`): a cheap, no-LLM scan
   of each step's reported output for failure signals. It is the only check
   needed for the ``stop`` policy and gates the more expensive supervisor.

2. **AI safety supervisor** (:class:`SafetySupervisor`): when a step trips the
   wire under the ``supervise`` policy, a separate LLM call adjudicates the
   outcome and decides how to recover *safely* — confirm success (false
   alarm), retry, mitigate with one corrective instruction, or abort. Abort
   (and any unrecoverable error / supervisor outage) triggers the caller's
   emergency stop. This is the "Claude is clever enough to mitigate the error"
   layer.

The runner is a generator that yields plain ``dict`` events so the Gradio layer
can render a live transcript without this module importing any UI code. It runs
fine without an LLM (``--simple`` mode): the supervisor reports itself
unavailable and ``supervise`` degrades to ``stop`` (fail-safe).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from ranger_llm_ui.scenarios import (
    POLICY_CONTINUE,
    POLICY_STOP,
    POLICY_SUPERVISE,
    Scenario,
)

logger = logging.getLogger(__name__)

# Recovery action verbs the supervisor may return.
ACTION_OK = "ok"
ACTION_RETRY = "retry"
ACTION_MITIGATE = "mitigate"
ACTION_ABORT = "abort"
_VALID_ACTIONS = {ACTION_OK, ACTION_RETRY, ACTION_MITIGATE, ACTION_ABORT}


# --- Heuristic error tripwire ---------------------------------------------
# High-precision signals: their presence almost always means a real failure.
_STRONG_ERROR_MARKERS = (
    "i encountered an error",
    "error:",
    "traceback",
    "exception:",
    "is not available",
    "is unavailable",
    "not initialized",
    "not initialised",
    "timed out",
    "request timed out",
    "❌",
    "⚠️",
    "request cancelled",
    "could not connect",
    "no such tool",
    "tool not found",
    # High-precision robot-action failure phrasings. These often arrive in a
    # polite, solution-offering sentence ("The pick failed — the object wasn't
    # detected…") that carries only one generic weak marker, so call them out
    # explicitly. Each is specific enough to rarely appear in a success report.
    "pick failed",
    "place failed",
    "grasp failed",
    "handover failed",
    "failed to pick",
    "failed to place",
    "failed to grasp",
    "failed to detect",
    "out of workspace",
    "out of the workspace",
    "out of reach",
    "wasn't detected",
    "was not detected",
    "not currently holding",
    "i'm not holding",
    "i am not holding",
    "need to pick up",
    "nothing to hand",
)
# Weaker signals: only treated as a failure when two or more co-occur, to keep
# benign phrasing ("I could not find any obstacles, all clear") from tripping.
_WEAK_ERROR_MARKERS = (
    "failed",
    "failure",
    "could not",
    "couldn't",
    "unable to",
    "cannot ",
    "can't ",
    "not found",
    "no response",
    "did not",
    "didn't",
    "denied",
    "refused",
    "aborted",
    "invalid",
)


def looks_like_error(output: str) -> bool:
    """Cheap, no-LLM heuristic: does this step output look like a failure?

    Empty output is treated as suspicious. Otherwise a single *strong* marker,
    or two or more *weak* markers, trips the wire. False positives are expected
    and are precisely what the :class:`SafetySupervisor` exists to adjudicate;
    under the ``stop`` policy they err on the side of halting (fail-safe).
    """
    if output is None or not output.strip():
        return True
    low = output.lower()
    if any(marker in low for marker in _STRONG_ERROR_MARKERS):
        return True
    weak_hits = sum(1 for marker in _WEAK_ERROR_MARKERS if marker in low)
    return weak_hits >= 2


def content_to_text(content: Any) -> str:
    """Flatten an LLM/agent content value (str | list of blocks | dict) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            else:
                parts.append(str(block))
        return " ".join(p for p in parts if p)
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    return str(content) if content is not None else ""


def _tool_names(result: dict) -> list[str]:
    """Extract tool names from an agent result's intermediate steps, if any."""
    names: list[str] = []
    for entry in result.get("intermediate_steps", []) or []:
        action = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
        name = getattr(action, "tool", None)
        if name:
            names.append(str(name))
    return names


@dataclass
class Verdict:
    """A safety supervisor's decision about a (possibly failed) step."""

    action: str             # ok | retry | mitigate | abort
    reason: str = ""        # short human-readable justification
    fix: str = ""           # corrective instruction (for retry/mitigate)


_SUPERVISOR_SYSTEM = """You are the SAFETY SUPERVISOR for an autonomous run of a \
Ranger garden robot. The robot executes a scenario as a list of natural-language \
steps, one at a time. After EACH step you are shown the step instruction and the \
robot's own report of what happened, and you judge whether the step actually \
SUCCEEDED and, if not, how to proceed SAFELY.

Read the robot's report carefully — it is often phrased politely even when the \
action did NOT happen. Treat as a FAILURE any report that: says an action failed \
or could not be done; says the target was not detected / out of reach / out of \
workspace; reports the robot is not holding an object it was supposed to hold; \
asks the operator a question or proposes alternatives INSTEAD of completing the \
step; or otherwise does not confirm the step was carried out. A status/lookup step \
(e.g. reporting a position or battery) succeeds if it returns the information.

Reply with ONLY a single JSON object, no prose, no code fences:
{"action": "ok|retry|mitigate|abort", "reason": "<short>", "fix": "<instruction or empty>"}

Definitions:
- "ok": the step genuinely succeeded. Continue. (Most steps succeed — use "ok" \
unless there is a clear sign the action did not happen.)
- "retry": a transient/benign problem; re-run the SAME step. Put a clearer \
rephrasing of the step in "fix", or leave "fix" empty to repeat it verbatim.
- "mitigate": the step failed but is safely recoverable. "fix" MUST be ONE \
concrete, conservative corrective robot instruction to run before re-attempting \
the step (e.g. "Move the arm to the ready pose", "Back up 0.3 meters and stop", \
"Reset odometry"). Never propose faster or larger motions to push through a failure.
- "abort": the situation is unsafe or not recoverable. Stop the scenario; the \
runner will trigger an emergency stop.

Safety rules: prioritize stopping over progress. If a movement or manipulation \
step failed in a way that could be unsafe, or you are unsure, choose "abort". \
Keep "reason" under 20 words."""


class SafetySupervisor:
    """LLM-backed adjudicator that decides how to recover from a failed step.

    Wraps a LangChain chat model (the same one the agent uses, obtained via
    ``RangerAgent.get_llm()``). When no model is available (e.g. ``--simple``
    mode), :meth:`available` returns ``False`` and the runner degrades the
    ``supervise`` policy to a fail-safe ``stop``.
    """

    def __init__(self, llm: Optional[Any]):
        self._llm = llm

    def available(self) -> bool:
        return self._llm is not None

    def judge(
        self,
        scenario_title: str,
        step: str,
        output: str,
        step_index: int = 0,
        total: int = 0,
        flagged: bool = False,
    ) -> Verdict:
        """Ask the LLM to judge a step and propose a safe recovery if it failed.

        ``flagged`` passes the cheap heuristic's opinion as a hint; the LLM is
        the authority and may overrule it in either direction.
        """
        if self._llm is None:
            return Verdict(ACTION_ABORT, "No safety supervisor LLM available.")

        hint = (
            "An automatic keyword check flagged this output as a POSSIBLE failure."
            if flagged
            else "An automatic keyword check did not flag this output, but judge it yourself."
        )
        human = (
            f"Scenario: {scenario_title}\n"
            f"Step {step_index + 1}"
            + (f" of {total}" if total else "")
            + f": {step}\n\n"
            f"Robot's report of the result:\n\"\"\"\n{output.strip()}\n\"\"\"\n\n"
            f"{hint}\n"
            "Decide: did this step succeed, and if not, how should the robot "
            "proceed safely? Respond with the JSON object only."
        )
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = self._llm.invoke(
                [
                    SystemMessage(content=_SUPERVISOR_SYSTEM),
                    HumanMessage(content=human),
                ]
            )
            text = content_to_text(getattr(response, "content", response))
            return self._parse_verdict(text)
        except Exception as e:  # pragma: no cover - network/LLM failure path
            logger.warning("Safety supervisor LLM call failed: %s", e)
            return Verdict(
                ACTION_ABORT, f"Supervisor unavailable ({e}); stopping for safety."
            )

    @staticmethod
    def _parse_verdict(text: str) -> Verdict:
        """Parse the supervisor's reply into a :class:`Verdict`, defensively.

        Falls back to keyword sniffing if strict JSON parsing fails, and to a
        fail-safe ``abort`` if even that is ambiguous (the supervisor is only
        consulted on already-suspicious steps).
        """
        text = (text or "").strip()
        # Try to locate the first {...} JSON object.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                action = str(data.get("action", "")).strip().lower()
                if action in _VALID_ACTIONS:
                    return Verdict(
                        action=action,
                        reason=str(data.get("reason", "")).strip(),
                        fix=str(data.get("fix", "")).strip(),
                    )
            except (json.JSONDecodeError, AttributeError):
                pass

        # Fallback: keyword sniff on the raw text.
        low = text.lower()
        for action in (ACTION_ABORT, ACTION_MITIGATE, ACTION_RETRY, ACTION_OK):
            if action in low:
                return Verdict(action, "Parsed from non-JSON supervisor reply.")
        return Verdict(
            ACTION_ABORT, "Could not parse supervisor reply; stopping for safety."
        )


def _event(kind: str, **data: Any) -> dict:
    data["kind"] = kind
    return data


def run_scenario(
    scenario: Scenario,
    *,
    invoke: Callable[[str], dict],
    emergency_stop: Callable[[], Any],
    supervisor: Optional[SafetySupervisor] = None,
    policy: str = POLICY_STOP,
    max_attempts: int = 2,
    stop_event: Optional[threading.Event] = None,
    pause_event: Optional[threading.Event] = None,
) -> Iterator[dict]:
    """Execute ``scenario`` step by step, yielding event dicts for the UI.

    Args:
        scenario: the parsed scenario to run.
        invoke: callable mapping a prompt string to an agent result dict
            (``{"output": ..., "intermediate_steps": [...]}`` — i.e.
            ``RangerAgent.invoke`` / ``SimpleAgent.invoke``).
        emergency_stop: callable invoked on abort to physically stop the robot.
        supervisor: the :class:`SafetySupervisor` (used only under ``supervise``).
        policy: one of ``stop`` / ``supervise`` / ``continue``.
        max_attempts: max recovery cycles per step before a fail-safe abort.
        stop_event: set externally to halt the run between steps/attempts.
        pause_event: set externally to pause the run between steps.

    Event ``kind`` values: ``scenario_start``, ``step_start``, ``step_result``,
    ``supervisor``, ``recovery``, ``info``, ``paused``, ``stopped``,
    ``aborted``, ``done``.
    """
    total = scenario.num_steps
    counters = {"ok": 0, "recovered": 0, "failed": 0}

    yield _event(
        "scenario_start",
        title=scenario.title,
        total=total,
        policy=policy,
    )

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    if total == 0:
        yield _event("info", text="This scenario has no steps to run.")
        yield _event("done", counters=counters, total=0, completed=True)
        return

    supervise = policy == POLICY_SUPERVISE
    supervisor_ok = supervise and supervisor is not None and supervisor.available()
    if supervise and not supervisor_ok:
        yield _event(
            "info",
            text=(
                "AI safety supervisor is unavailable (no LLM); falling back to "
                "**Stop on error** for safety."
            ),
        )

    for idx, step in enumerate(scenario.steps):
        # --- pause / stop gate between steps -------------------------------
        while pause_event is not None and pause_event.is_set() and not _stopped():
            yield _event("paused", index=idx, total=total)
            time.sleep(0.3)
        if _stopped():
            yield _event("stopped", index=idx, total=total, counters=counters)
            return

        original = step
        prompt = step
        attempt = 0  # 0 = first try; >0 = recovery cycles
        outcome: Optional[str] = None  # ok | recovered | failed | abort
        abort_reason = ""

        while True:
            # Honor pause/stop between recovery attempts too, not only between
            # whole steps — otherwise a paused run would keep retrying.
            while pause_event is not None and pause_event.is_set() and not _stopped():
                yield _event("paused", index=idx, total=total)
                time.sleep(0.3)
            if _stopped():
                yield _event("stopped", index=idx, total=total, counters=counters)
                return

            yield _event(
                "step_start",
                index=idx,
                total=total,
                prompt=prompt,
                attempt=attempt,
            )

            try:
                result = invoke(prompt)
            except Exception as e:  # invoke should not raise, but be safe
                logger.error("Scenario step invoke raised: %s", e)
                result = {"output": f"I encountered an error: {e}",
                          "intermediate_steps": []}

            output = content_to_text(result.get("output", ""))
            tools = _tool_names(result)
            flagged = looks_like_error(output)

            yield _event(
                "step_result",
                index=idx,
                total=total,
                prompt=prompt,
                output=output,
                tools=tools,
                flagged=flagged,
                attempt=attempt,
                recovery=attempt > 0,
            )

            # In supervise mode the LLM is the authority and judges EVERY step
            # (not just heuristic-flagged ones) — polite failure reports like
            # "the pick failed, want me to try PickAt?" carry no strong marker
            # and would otherwise slip through as success. The heuristic gate is
            # only used for the no-LLM stop/continue paths.
            if not supervisor_ok:
                if not flagged:
                    outcome = "recovered" if attempt > 0 else "ok"
                    break
                if policy == POLICY_CONTINUE:
                    yield _event(
                        "info",
                        text="Issue detected, but policy is **Run all** — continuing.",
                    )
                    outcome = "failed"
                    break
                # stop policy, or supervise requested but no LLM available
                outcome = "abort"
                abort_reason = (
                    "Step reported an error (Stop-on-error policy)."
                    if policy == POLICY_STOP
                    else "Step reported an error and no AI supervisor is available."
                )
                break

            # --- supervise: the LLM judges this step and how to recover -----
            verdict = supervisor.judge(
                scenario.title, original, output,
                step_index=idx, total=total, flagged=flagged,
            )
            yield _event(
                "supervisor",
                index=idx,
                action=verdict.action,
                reason=verdict.reason,
                fix=verdict.fix,
                flagged=flagged,
            )

            if verdict.action == ACTION_OK:
                outcome = "recovered" if attempt > 0 else "ok"
                break
            if verdict.action == ACTION_ABORT:
                outcome = "abort"
                abort_reason = verdict.reason or "Safety supervisor requested abort."
                break
            if attempt >= max_attempts:
                yield _event(
                    "info",
                    text=(
                        f"Reached max recovery attempts ({max_attempts}); "
                        "aborting for safety."
                    ),
                )
                outcome = "abort"
                abort_reason = f"No recovery after {max_attempts} attempt(s)."
                break

            # Perform the recovery, then loop to re-run the step.
            attempt += 1
            if verdict.action == ACTION_MITIGATE and verdict.fix:
                yield _event(
                    "recovery",
                    kind_label="mitigate",
                    index=idx,
                    prompt=verdict.fix,
                    attempt=attempt,
                )
                if _stopped():
                    yield _event("stopped", index=idx, total=total, counters=counters)
                    return
                try:
                    fix_result = invoke(verdict.fix)
                except Exception as e:  # pragma: no cover - defensive
                    fix_result = {"output": f"I encountered an error: {e}",
                                  "intermediate_steps": []}
                fix_output = content_to_text(fix_result.get("output", ""))
                fix_failed = looks_like_error(fix_output)
                yield _event(
                    "step_result",
                    index=idx,
                    total=total,
                    prompt=verdict.fix,
                    output=fix_output,
                    tools=_tool_names(fix_result),
                    flagged=fix_failed,
                    attempt=attempt,
                    recovery=True,
                    mitigation=True,
                )
                # Fail-safe: if the corrective action ITSELF failed, the robot is
                # not in the safe precondition state the supervisor approved the
                # retry under — abort rather than blindly re-run the step.
                if fix_failed:
                    outcome = "abort"
                    abort_reason = (
                        "Corrective (mitigation) action failed; stopping for safety."
                    )
                    break
                prompt = original  # re-attempt the original step next
            else:  # retry (optionally rephrased)
                prompt = verdict.fix or original
                yield _event(
                    "recovery",
                    kind_label="retry",
                    index=idx,
                    prompt=prompt,
                    attempt=attempt,
                )
            # loop continues -> re-runs `prompt`

        # --- step finished one way or another ------------------------------
        if outcome == "abort":
            counters["failed"] += 1
            try:
                emergency_stop()
            except Exception as e:  # pragma: no cover - defensive
                logger.error("Emergency stop during scenario abort failed: %s", e)
            yield _event(
                "aborted",
                index=idx,
                total=total,
                reason=abort_reason,
                counters=dict(counters),
            )
            return

        counters[outcome] += 1
        yield _event(
            "step_done",
            index=idx,
            total=total,
            outcome=outcome,
            counters=dict(counters),
        )

    yield _event("done", counters=dict(counters), total=total, completed=True)
