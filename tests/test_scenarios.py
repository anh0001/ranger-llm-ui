"""
Tests for the scenario loader, safety supervisor, and scenario runner.

These exercise the feature end-to-end without ROS 2, hardware, or a real LLM:
the agent is replaced by simple stub callables and the safety supervisor by a
fake chat model, so the safety logic (stop / supervise / continue, retry /
mitigate / abort, pause / stop) is verified deterministically.
"""

import threading
from unittest.mock import Mock

import pytest

from ranger_llm_ui.scenarios import (
    Scenario,
    parse_scenario_text,
    load_scenarios,
    scenarios_dir,
    POLICY_STOP,
    POLICY_SUPERVISE,
    POLICY_CONTINUE,
    POLICY_LABELS,
    LABEL_TO_POLICY,
    policy_label,
)
from ranger_llm_ui.scenario_runner import (
    looks_like_error,
    content_to_text,
    SafetySupervisor,
    run_scenario,
    ACTION_OK,
    ACTION_ABORT,
    ACTION_MITIGATE,
)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
class TestParseScenario:
    def test_metadata_and_steps(self):
        text = (
            "# title: My Patrol\n"
            "# description: Do a thing\n"
            "# safety: supervise\n"
            "# fresh_context: false\n"
            "# just a comment\n"
            "\n"
            "Move forward 1 meter.\n"
            "  Turn left 90 degrees.  \n"
            "\n"
            "Stop.\n"
        )
        sc = parse_scenario_text(text, name="my_patrol")
        assert sc.title == "My Patrol"
        assert sc.description == "Do a thing"
        assert sc.safety == POLICY_SUPERVISE
        assert sc.fresh_context is False
        assert sc.steps == [
            "Move forward 1 meter.",
            "Turn left 90 degrees.",
            "Stop.",
        ]
        assert sc.num_steps == 3
        assert sc.name == "my_patrol"

    def test_defaults_when_no_metadata(self):
        sc = parse_scenario_text("Do step one.\nDo step two.\n", name="garden_patrol")
        # Title falls back to a prettified file stem.
        assert sc.title == "Garden Patrol"
        assert sc.description == ""
        assert sc.safety == POLICY_STOP  # safe default
        assert sc.fresh_context is True
        assert sc.num_steps == 2

    def test_fresh_context_keeps_default_on_unrecognized(self):
        # Empty / unknown values must NOT silently flip the documented default.
        assert parse_scenario_text("# fresh_context:\nx").fresh_context is True
        assert parse_scenario_text("# fresh_context: enabled\nx").fresh_context is True
        assert parse_scenario_text("# fresh_context: false\nx").fresh_context is False
        assert parse_scenario_text("# fresh_context: no\nx").fresh_context is False
        assert parse_scenario_text("# fresh_context: TRUE \nx").fresh_context is True

    def test_safety_aliases(self):
        assert parse_scenario_text("# safety: mitigate\nx").safety == POLICY_SUPERVISE
        assert parse_scenario_text("# safety: ignore\nx").safety == POLICY_CONTINUE
        assert parse_scenario_text("# safety: safe\nx").safety == POLICY_STOP
        # Unknown value keeps the default.
        assert parse_scenario_text("# safety: bogus\nx").safety == POLICY_STOP

    def test_blank_and_comment_only_has_no_steps(self):
        sc = parse_scenario_text("# title: Empty\n# only comments\n\n   \n")
        assert sc.num_steps == 0

    def test_raw_preserved(self):
        text = "# title: Keep\nStep A.\n"
        sc = parse_scenario_text(text)
        assert sc.raw == text


class TestPolicyLabels:
    def test_label_roundtrip(self):
        for key, label in POLICY_LABELS.items():
            assert LABEL_TO_POLICY[label] == key
        assert policy_label(POLICY_SUPERVISE) == POLICY_LABELS[POLICY_SUPERVISE]
        # Unknown policy returns the safe-default label.
        assert policy_label("nonsense") == POLICY_LABELS[POLICY_STOP]


# --------------------------------------------------------------------------
# Bundled scenarios on disk
# --------------------------------------------------------------------------
class TestBundledScenarios:
    def test_repo_scenarios_load(self):
        # The repo ships scenarios/ next to the package.
        directory = scenarios_dir()
        assert directory is not None, "scenarios directory should be discoverable"
        loaded = load_scenarios(directory)
        assert len(loaded) >= 5
        titles = {s.title for s in loaded}
        assert "Welcome & Self-Check" in titles
        for sc in loaded:
            assert sc.num_steps > 0
            assert sc.safety in POLICY_LABELS

    def test_load_missing_dir_is_empty(self, tmp_path):
        assert load_scenarios(tmp_path / "does_not_exist") == []


# --------------------------------------------------------------------------
# Heuristic error tripwire
# --------------------------------------------------------------------------
class TestLooksLikeError:
    @pytest.mark.parametrize("text", [
        "",
        "   ",
        "I encountered an error: boom",
        "ERROR: tool not found",
        "The service is not available right now.",
        "Request timed out after 60 seconds.",
        "Failed to connect, could not reach the server",  # 2 weak markers
    ])
    def test_flags_failures(self, text):
        assert looks_like_error(text) is True

    @pytest.mark.parametrize("text", [
        "Battery is at 87 percent and healthy.",
        "I could not find any obstacles, all clear.",  # single weak marker
        "Done. I moved forward 1 meter as requested.",
        "All systems nominal.",
    ])
    def test_passes_benign(self, text):
        assert looks_like_error(text) is False


class TestContentToText:
    def test_variants(self):
        assert content_to_text("hi") == "hi"
        assert content_to_text([{"type": "text", "text": "a"}, {"text": "b"}]) == "a b"
        assert content_to_text({"text": "z"}) == "z"
        assert content_to_text(None) == ""


# --------------------------------------------------------------------------
# Safety supervisor
# --------------------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Minimal chat-model stand-in: returns a fixed content for .invoke()."""

    def __init__(self, content):
        self._content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _FakeMessage(self._content)


class TestSafetySupervisor:
    def test_unavailable_without_llm(self):
        sup = SafetySupervisor(None)
        assert sup.available() is False
        v = sup.judge("t", "step", "output")
        assert v.action == ACTION_ABORT

    def test_parses_json_verdict(self):
        llm = _FakeLLM('{"action": "mitigate", "reason": "recoverable", "fix": "Back up"}')
        sup = SafetySupervisor(llm)
        assert sup.available() is True
        v = sup.judge("Patrol", "Move forward", "I encountered an error")
        assert v.action == ACTION_MITIGATE
        assert v.fix == "Back up"
        assert "recoverable" in v.reason

    def test_parses_json_inside_prose(self):
        llm = _FakeLLM('Sure! {"action":"ok","reason":"false alarm","fix":""} done')
        v = SafetySupervisor(llm).judge("t", "s", "o")
        assert v.action == ACTION_OK

    def test_fallback_keyword_sniff(self):
        llm = _FakeLLM("I think we should abort this run.")
        v = SafetySupervisor(llm).judge("t", "s", "o")
        assert v.action == ACTION_ABORT

    def test_unparseable_defaults_to_abort(self):
        llm = _FakeLLM("completely unrelated text with no decision")
        v = SafetySupervisor(llm).judge("t", "s", "o")
        assert v.action == ACTION_ABORT


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def _ok(_prompt):
    return {"output": "Done, all good.", "intermediate_steps": []}


def _kinds(events):
    return [e["kind"] for e in events]


class TestRunScenario:
    def test_all_ok_completes(self):
        sc = Scenario("t", "T", "", ["a", "b", "c"])
        estop = Mock()
        events = list(run_scenario(
            sc, invoke=_ok, emergency_stop=estop, policy=POLICY_STOP,
        ))
        kinds = _kinds(events)
        assert kinds[0] == "scenario_start"
        assert kinds[-1] == "done"
        done = events[-1]
        assert done["counters"] == {"ok": 3, "recovered": 0, "failed": 0}
        estop.assert_not_called()
        # One step_done per step.
        assert sum(1 for k in kinds if k == "step_done") == 3

    def test_stop_policy_aborts_and_estops(self):
        def invoke(prompt):
            if "boom" in prompt:
                return {"output": "I encountered an error: boom", "intermediate_steps": []}
            return _ok(prompt)

        sc = Scenario("t", "T", "", ["fine", "boom", "never"])
        estop = Mock()
        events = list(run_scenario(
            sc, invoke=invoke, emergency_stop=estop, policy=POLICY_STOP,
        ))
        kinds = _kinds(events)
        assert "aborted" in kinds
        assert "done" not in kinds
        estop.assert_called_once()
        aborted = next(e for e in events if e["kind"] == "aborted")
        assert aborted["index"] == 1  # the "boom" step
        # The third step never runs.
        starts = [e for e in events if e["kind"] == "step_start"]
        assert len(starts) == 2

    def test_continue_policy_runs_through_errors(self):
        def invoke(prompt):
            if "boom" in prompt:
                return {"output": "Failed: could not do it", "intermediate_steps": []}
            return _ok(prompt)

        sc = Scenario("t", "T", "", ["boom", "fine"])
        estop = Mock()
        events = list(run_scenario(
            sc, invoke=invoke, emergency_stop=estop, policy=POLICY_CONTINUE,
        ))
        assert _kinds(events)[-1] == "done"
        estop.assert_not_called()
        assert events[-1]["counters"]["failed"] == 1
        assert events[-1]["counters"]["ok"] == 1

    def test_supervise_ok_is_false_alarm(self):
        # Heuristic flags it, supervisor says it actually succeeded.
        def invoke(prompt):
            return {"output": "Could not, did not — but actually fine",
                    "intermediate_steps": []}

        sc = Scenario("t", "T", "", ["a"])
        estop = Mock()
        llm = _FakeLLM('{"action":"ok","reason":"false alarm","fix":""}')
        events = list(run_scenario(
            sc, invoke=invoke, emergency_stop=estop, policy=POLICY_SUPERVISE,
            supervisor=SafetySupervisor(llm),
        ))
        kinds = _kinds(events)
        assert "supervisor" in kinds
        assert kinds[-1] == "done"
        estop.assert_not_called()
        assert events[-1]["counters"]["ok"] == 1

    def test_supervise_mitigate_then_recovers(self):
        state = {"mitigated": False}

        def invoke(prompt):
            if prompt == "Back up 0.3 meters":
                state["mitigated"] = True
                return _ok(prompt)
            if "boom" in prompt and not state["mitigated"]:
                return {"output": "I encountered an error: boom",
                        "intermediate_steps": []}
            return _ok(prompt)

        sc = Scenario("t", "T", "", ["boom step"])
        estop = Mock()
        llm = _FakeLLM('{"action":"mitigate","reason":"recoverable","fix":"Back up 0.3 meters"}')
        events = list(run_scenario(
            sc, invoke=invoke, emergency_stop=estop, policy=POLICY_SUPERVISE,
            supervisor=SafetySupervisor(llm), max_attempts=2,
        ))
        kinds = _kinds(events)
        assert "recovery" in kinds
        assert kinds[-1] == "done"
        estop.assert_not_called()
        assert events[-1]["counters"]["recovered"] == 1

    def test_supervise_mitigation_failure_aborts(self):
        # The corrective action ITSELF fails -> fail-safe abort (don't blindly
        # re-run the original step).
        def invoke(prompt):
            if prompt == "Back up":
                return {"output": "I encountered an error: cannot back up",
                        "intermediate_steps": []}
            return {"output": "I encountered an error: boom", "intermediate_steps": []}

        sc = Scenario("t", "T", "", ["boom step"])
        estop = Mock()
        llm = _FakeLLM('{"action":"mitigate","reason":"recoverable","fix":"Back up"}')
        events = list(run_scenario(
            sc, invoke=invoke, emergency_stop=estop, policy=POLICY_SUPERVISE,
            supervisor=SafetySupervisor(llm), max_attempts=3,
        ))
        assert _kinds(events)[-1] == "aborted"
        estop.assert_called_once()
        aborted = next(e for e in events if e["kind"] == "aborted")
        assert "mitigation" in aborted["reason"].lower()
        # Original step is NOT re-run after the failed mitigation: invoke is
        # called for the step (boom) + the mitigation (Back up) only.
        starts = [e for e in events if e["kind"] == "step_start"]
        assert len(starts) == 1  # only the initial attempt; no post-mitigation retry

    def test_supervise_abort_triggers_estop(self):
        def invoke(prompt):
            return {"output": "I encountered an error", "intermediate_steps": []}

        sc = Scenario("t", "T", "", ["a"])
        estop = Mock()
        llm = _FakeLLM('{"action":"abort","reason":"unsafe","fix":""}')
        events = list(run_scenario(
            sc, invoke=invoke, emergency_stop=estop, policy=POLICY_SUPERVISE,
            supervisor=SafetySupervisor(llm),
        ))
        assert _kinds(events)[-1] == "aborted"
        estop.assert_called_once()

    def test_supervise_without_llm_falls_back_to_stop(self):
        def invoke(prompt):
            return {"output": "I encountered an error", "intermediate_steps": []}

        sc = Scenario("t", "T", "", ["a"])
        estop = Mock()
        events = list(run_scenario(
            sc, invoke=invoke, emergency_stop=estop, policy=POLICY_SUPERVISE,
            supervisor=SafetySupervisor(None),
        ))
        kinds = _kinds(events)
        assert "aborted" in kinds
        # An info event warns about the fallback.
        assert any(e["kind"] == "info" for e in events)
        estop.assert_called_once()

    def test_max_attempts_exhausted_aborts(self):
        # Supervisor keeps asking to retry but the step never recovers.
        def invoke(prompt):
            return {"output": "I encountered an error", "intermediate_steps": []}

        sc = Scenario("t", "T", "", ["a"])
        estop = Mock()
        llm = _FakeLLM('{"action":"retry","reason":"transient","fix":""}')
        events = list(run_scenario(
            sc, invoke=invoke, emergency_stop=estop, policy=POLICY_SUPERVISE,
            supervisor=SafetySupervisor(llm), max_attempts=2,
        ))
        assert _kinds(events)[-1] == "aborted"
        estop.assert_called_once()
        # Initial attempt + 2 retries = 3 step_start events.
        assert sum(1 for e in events if e["kind"] == "step_start") == 3

    def test_stop_event_halts_between_steps(self):
        stop = threading.Event()
        calls = {"n": 0}

        def invoke(prompt):
            calls["n"] += 1
            stop.set()  # request stop after the first step runs
            return _ok(prompt)

        sc = Scenario("t", "T", "", ["a", "b", "c"])
        estop = Mock()
        events = list(run_scenario(
            sc, invoke=invoke, emergency_stop=estop, policy=POLICY_STOP,
            stop_event=stop,
        ))
        assert _kinds(events)[-1] == "stopped"
        assert calls["n"] == 1  # only the first step ran

    def test_empty_scenario(self):
        sc = Scenario("t", "T", "", [])
        events = list(run_scenario(sc, invoke=_ok, emergency_stop=Mock()))
        kinds = _kinds(events)
        assert kinds[-1] == "done"
        assert events[-1]["total"] == 0

    def test_invoke_exception_is_treated_as_error(self):
        def invoke(prompt):
            raise RuntimeError("kaboom")

        sc = Scenario("t", "T", "", ["a"])
        estop = Mock()
        events = list(run_scenario(
            sc, invoke=invoke, emergency_stop=estop, policy=POLICY_STOP,
        ))
        # The raised error is caught, flagged, and aborts under stop policy.
        assert _kinds(events)[-1] == "aborted"
        estop.assert_called_once()
