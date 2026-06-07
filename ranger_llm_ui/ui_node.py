"""
UI Node - Main entry point for the Ranger LLM UI.

This module starts the ROS 2 node and Gradio web interface. It provides:
- A chat interface for natural language robot control
- Emergency stop button
- Status display (battery, etc.)
- Manual teleop controls (optional)

Usage:
    ros2 run ranger_llm_ui ui_node
    # Or directly:
    python -m ranger_llm_ui.ui_node
"""

import os
import re
import sys
import asyncio
import threading
import logging
import signal
from typing import Optional, Generator, Any
from pathlib import Path
from functools import wraps

import gradio as gr
from dotenv import load_dotenv

from ranger_llm_ui.agent_interface import create_agent, RangerAgent, LLMProvider
from ranger_llm_ui.tools.movement_tools import get_ros_interface
from ranger_llm_ui.tools.status_tools import get_status_interface
from ranger_llm_ui.tools.camera_tools import get_camera_interface
from ranger_llm_ui.utils.logger import setup_logging, get_command_logger
from ranger_llm_ui import scenarios as scenario_lib
from ranger_llm_ui.scenarios import (
    load_scenarios,
    parse_scenario_text,
    POLICY_LABELS,
    LABEL_TO_POLICY,
    POLICY_STOP,
    POLICY_SUPERVISE,
)
from ranger_llm_ui.scenario_runner import run_scenario, SafetySupervisor, content_to_text
from ranger_llm_ui.voice import (
    get_transcriber,
    get_synthesizer,
    get_corrector,
    voice_status,
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning("ROS 2 (rclpy) not available. Running in simulation mode.")

# Try to import the movement action server (requires ranger_llm_msgs to be built)
try:
    from ranger_llm_ui.movement_action_server import MovementActionServer
    ACTION_SERVER_AVAILABLE = True
except ImportError:
    ACTION_SERVER_AVAILABLE = False
    if ROS_AVAILABLE:
        logger.warning(
            "Movement action server not available. "
            "Build ranger_llm_msgs first: colcon build --packages-select ranger_llm_msgs"
        )


class RangerUINode:
    """
    ROS 2 node for the Ranger LLM UI.

    This node manages the Gradio interface and ROS 2 communication.
    """

    def __init__(
        self,
        node_name: str = "ranger_llm_ui",
        llm_provider: str = "openai",
        model_name: Optional[str] = None,
        server_port: int = 7860,
        share: bool = False,
        simple_mode: bool = False,
        debug_mode: bool = False,
    ):
        """
        Initialize the UI node.

        Args:
            node_name: ROS 2 node name
            llm_provider: LLM provider (openai, ollama, anthropic)
            model_name: Model name (optional)
            server_port: Gradio server port
            share: Create a public Gradio link
            simple_mode: Use simple agent without LLM
            debug_mode: Use only ROSA base prompts, skip Ranger-specific prompts
        """
        self.node_name = node_name
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.server_port = server_port
        self.share = share
        self.simple_mode = simple_mode
        self.debug_mode = debug_mode

        self._node: Optional[Any] = None
        self._executor: Optional[Any] = None
        self._spin_thread: Optional[threading.Thread] = None

        # Initialize agent (will be set up after ROS node is created)
        self.agent: Optional[RangerAgent] = None

        # Command logger
        self.logger = get_command_logger()

        # Cancellation flag for stopping long-running requests
        self._cancel_requested = threading.Event()

        # Scenario runner control flags (stop = halt run; pause = hold between
        # steps). Both are honored by run_scenario() between steps/attempts.
        self._scenario_stop = threading.Event()
        self._scenario_pause = threading.Event()
        # True while a scenario run is streaming (re-entrancy guard).
        self._scenario_active = False
        # Loaded lazily in create_ui(); list[Scenario].
        self._scenarios: list = []

        # Serializes every agent.invoke() call (chat, scenario, manual teleop)
        # so the single ROSA executor / shared chat_history is never driven by
        # two Gradio events at once (different events run in independent
        # concurrency groups, so the queue does not serialize them for us).
        self._agent_lock = threading.Lock()

    def initialize_ros(self):
        """Initialize ROS 2 node if available."""
        if not ROS_AVAILABLE:
            logger.info("Running without ROS 2 (simulation mode)")
            return

        try:
            rclpy.init()
            self._node = rclpy.create_node(self.node_name)
            logger.info(f"ROS 2 node '{self.node_name}' initialized")

            # Start spinning UI node in background thread
            self._executor = MultiThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
            self._spin_thread.start()

            # Start movement action server in its own executor/thread.
            # It must NOT share an executor with the action client (UI node)
            # to avoid duplicate goal ID errors in ROS 2 Humble.
            self._movement_server = None
            self._movement_executor = None
            self._movement_spin_thread = None
            if ACTION_SERVER_AVAILABLE:
                try:
                    self._movement_server = MovementActionServer()
                    self._movement_executor = MultiThreadedExecutor()
                    self._movement_executor.add_node(self._movement_server)
                    self._movement_spin_thread = threading.Thread(
                        target=self._spin_movement_server, daemon=True
                    )
                    self._movement_spin_thread.start()
                    logger.info("Movement action server started in dedicated executor")
                except Exception as e:
                    logger.warning(f"Failed to start movement action server: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize ROS 2: {e}")
            self._node = None

    def _spin_ros(self):
        """Spin ROS 2 UI node in background."""
        try:
            self._executor.spin()
        except Exception as e:
            logger.error(f"ROS spin error: {e}")

    def _spin_movement_server(self):
        """Spin movement action server in its own background thread."""
        try:
            self._movement_executor.spin()
        except Exception as e:
            logger.error(f"Movement server spin error: {e}")

    def initialize_agent(self):
        """Initialize the LangChain agent."""
        try:
            self.agent = create_agent(
                provider=self.llm_provider,
                model_name=self.model_name,
                ros_node=self._node,
                simple_mode=self.simple_mode,
                debug_mode=self.debug_mode,
            )
            mode_label = "DEBUG (ROSA only)" if self.debug_mode else "Running (Ranger+ROSA)"
            logger.info(f"Agent initialized with provider: {self.llm_provider}, mode: {mode_label}")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            # Fall back to simple mode
            logger.info("Falling back to simple mode")
            self.agent = create_agent(ros_node=self._node, simple_mode=True)

    def set_debug_mode(self, enabled: bool) -> str:
        """Switch between Running and Debug mode and reinitialize the agent."""
        self.debug_mode = enabled
        self.initialize_agent()
        if enabled:
            return "Switched to **Debug mode** (ROSA base prompts only, no Ranger persona)"
        return "Switched to **Running mode** (Ranger + ROSA prompts active)"

    def set_model(self, model_choice: str) -> str:
        """Switch model and reinitialize the agent."""
        self.model_name = model_choice or None
        self.initialize_agent()
        return f"Model switched to **{self.model_name or 'provider default'}** (provider: {self.llm_provider})"

    def emergency_stop(self) -> str:
        """Execute emergency stop.

        Also halts any in-flight scenario run (the runner checks this flag
        between steps) so a single E-stop press from any tab stops both manual
        and scenario-driven motion.
        """
        self._scenario_stop.set()
        self._scenario_pause.clear()
        ros = get_ros_interface()
        ros.stop()
        logger.warning("EMERGENCY STOP executed")
        return "EMERGENCY STOP executed - Robot stopped"

    def get_battery_status(self) -> str:
        """Get current battery status for display."""
        interface = get_status_interface()
        level, status, _voltage = interface.get_battery_level()
        if level < 0:
            return "Battery: Unknown"
        return f"Battery: {level:.0f}% ({status})"

    def get_camera_image(self):
        """Get the latest camera image for display."""
        interface = get_camera_interface()
        return interface.get_latest_image()

    def cancel_chat(self):
        """Cancel the ongoing chat request."""
        self._cancel_requested.set()
        logger.info("Chat cancellation requested")
        return gr.update(visible=True), gr.update(visible=False)

    def _invoke_agent(self, prompt: str) -> dict:
        """Run a single agent.invoke() under the agent lock.

        All invoke paths (chat, scenario, manual teleop) funnel through here so
        the agent's single executor / chat_history is never driven concurrently
        by two Gradio events.
        """
        if self.agent is None:
            return {"output": "Agent not initialized.", "intermediate_steps": []}
        with self._agent_lock:
            return self.agent.invoke(prompt)

    def chat_response(
        self, message: str, history: list[dict]
    ) -> Generator[list[dict], None, None]:
        """
        Generate chat response with streaming.

        Args:
            message: User message
            history: Chat history as list of message dicts with 'role' and 'content'

        Yields:
            Updated history with streaming response
        """
        if not self.agent:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "Agent not initialized. Please check configuration."})
            yield history
            return

        # Add user message to history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})

        # Clear cancel flag at start of new request
        self._cancel_requested.clear()

        try:
            # For synchronous response (non-streaming) with timeout
            # Set timeout to 60 seconds to prevent hanging
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._invoke_agent, message)

                # Poll for cancellation or completion (max 60 seconds)
                elapsed_time = 0
                max_timeout = 60
                while elapsed_time < max_timeout:
                    try:
                        result = future.result(timeout=1)  # Check every 1 second
                        output = self._content_to_text(
                            result.get("output", "I couldn't process that request.")
                        )
                        break
                    except concurrent.futures.TimeoutError:
                        elapsed_time += 1
                        # Check if user requested cancellation
                        if self._cancel_requested.is_set():
                            logger.info("Chat request cancelled by user")
                            history[-1]["content"] = "❌ Request cancelled by user."
                            yield history
                            return
                        # Continue waiting if not cancelled and not timed out
                        continue
                else:
                    # Loop exited due to timeout
                    history[-1]["content"] = "⚠️ Request timed out after 60 seconds. The LLM is taking too long to respond. Please try a simpler command or check your LLM connection."
                    yield history
                    return

            # Check for intermediate steps to show tool usage
            intermediate_steps = result.get("intermediate_steps", [])
            if intermediate_steps:
                tool_info = []
                for action, observation in intermediate_steps:
                    tool_name = action.tool if hasattr(action, 'tool') else str(action)
                    tool_info.append(f"Used tool: {tool_name}")

                if tool_info:
                    output = "\n".join(tool_info) + "\n\n" + output

            if os.getenv("SHOW_LLM_USAGE", "").lower() in {"1", "true", "yes", "on"}:
                usage = result.get("usage") or {}
                total_tokens = usage.get("total_tokens")
                if total_tokens:
                    cost = usage.get("total_cost_usd")
                    cost_str = f", cost ${cost:.6f}" if isinstance(cost, (int, float)) else ""
                    output = (
                        f"{output}\n\n---\n"
                        f"Tokens: prompt {usage.get('prompt_tokens', 0)}, "
                        f"completion {usage.get('completion_tokens', 0)}, total {total_tokens}{cost_str}"
                    )

            history[-1]["content"] = output
            yield history

        except Exception as e:
            logger.error(f"Chat error: {e}")
            history[-1]["content"] = f"Error: {str(e)}"
            yield history

    def transcribe_audio(self, audio_path: Optional[str]) -> str:
        """Transcribe recorded mic audio to text for the command box.

        Runs an optional LLM second pass (gated on Whisper confidence, with a
        hard stop-command bypass) to fix obvious ASR errors against the robot
        command vocabulary before the text goes to the agent.
        """
        if not audio_path:
            return ""
        transcriber = get_transcriber()
        text = transcriber.transcribe(audio_path)
        logger.info("Voice command transcribed: %r", text)
        corrected = get_corrector().correct(
            text,
            transcriber.last_stats,
            provider=self.llm_provider,
            model=self.model_name,
        )
        if corrected != text:
            logger.info("Voice command corrected: %r", corrected)
        return corrected

    @staticmethod
    def _content_to_text(content) -> str:
        """Flatten chat content (str | list of blocks | dict | None) to text.

        Delegates to the shared :func:`scenario_runner.content_to_text` so the
        chat/TTS path and the scenario/supervisor path can never drift apart in
        how they coerce claude_proxy / Anthropic list-of-content-block replies.
        """
        return content_to_text(content)

    def synthesize_response(self, history: list[dict], enabled: bool):
        """Speak the latest assistant reply when voice output is enabled."""
        if not enabled or not history:
            return None
        last = history[-1]
        if not isinstance(last, dict) or last.get("role") != "assistant":
            return None
        text = self._content_to_text(last.get("content", ""))
        return get_synthesizer().synthesize(text)

    def teleop_forward(self, distance: float = 0.5) -> str:
        """Manual forward movement."""
        if self.agent:
            result = self._invoke_agent(f"move forward {distance} meters")
            return result.get("output", "Command sent")
        return "Agent not initialized"

    def teleop_backward(self, distance: float = 0.5) -> str:
        """Manual backward movement."""
        if self.agent:
            result = self._invoke_agent(f"move backward {distance} meters")
            return result.get("output", "Command sent")
        return "Agent not initialized"

    def teleop_left(self, angle: float = 45) -> str:
        """Manual left turn."""
        if self.agent:
            result = self._invoke_agent(f"turn left {angle} degrees")
            return result.get("output", "Command sent")
        return "Agent not initialized"

    def teleop_right(self, angle: float = 45) -> str:
        """Manual right turn."""
        if self.agent:
            result = self._invoke_agent(f"turn right {angle} degrees")
            return result.get("output", "Command sent")
        return "Agent not initialized"

    # ---------------------------------------------------------------------
    # Scenario runner (Scenarios tab)
    # ---------------------------------------------------------------------
    # Icons used to render per-step state in the steps preview.
    _STEP_ICONS = {
        "pending": "⬜",
        "running": "🔵",
        "ok": "✅",
        "recovered": "♻️",
        "failed": "❌",
    }

    def _scenario_titles(self) -> list[str]:
        return [s.title for s in self._scenarios]

    def _find_scenario(self, title: Optional[str]):
        for s in self._scenarios:
            if s.title == title:
                return s
        return None

    def _steps_view_md(self, scenario, current_idx: int, statuses: list[str]) -> str:
        """Render the numbered step list with per-step status icons."""
        if scenario is None or scenario.num_steps == 0:
            return "_No steps to show._"
        lines = []
        for i, step in enumerate(scenario.steps):
            st = statuses[i] if i < len(statuses) else "pending"
            icon = self._STEP_ICONS.get(st, "⬜")
            text = step if len(step) <= 100 else step[:97] + "…"
            text = text.replace("<", "&lt;").replace(">", "&gt;")
            if st == "running":
                text = f"**{text}**"
            lines.append(f"{icon} {i + 1}. {text}")
        # Two trailing spaces force a hard line break in Markdown.
        return "  \n".join(lines)

    def _scenario_status_md(
        self,
        total: int,
        counters: dict,
        state: str,
        current: Optional[int] = None,
    ) -> str:
        """Render the progress bar + status badge for a scenario run."""
        finished = counters.get("ok", 0) + counters.get("recovered", 0) + counters.get("failed", 0)
        pct = int(round(100 * finished / total)) if total else 0
        color = {
            "running": "#2563eb",
            "paused": "#d97706",
            "stopped": "#6b7280",
            "aborted": "#dc2626",
            "done": "#16a34a",
            "idle": "#9ca3af",
        }.get(state, "#2563eb")
        badge = {
            "running": "▶ running",
            "paused": "⏸ paused",
            "stopped": "⏹ stopped",
            "aborted": "🛑 aborted",
            "done": "✅ complete",
            "idle": "ready",
        }.get(state, state)
        if current is not None and current >= 0 and state in ("running", "paused"):
            where = f"step {current + 1} / {total}"
        else:
            where = f"{finished} / {total} steps"
        bar = (
            '<div style="background:#e5e7eb;border-radius:8px;height:12px;'
            'width:100%;overflow:hidden;margin:6px 0;">'
            f'<div style="background:{color};height:12px;width:{pct}%;'
            'transition:width .2s;"></div></div>'
        )
        legend = (
            f'✅ {counters.get("ok", 0)} ok · '
            f'♻️ {counters.get("recovered", 0)} recovered · '
            f'❌ {counters.get("failed", 0)} failed'
        )
        return f"**{badge}** · {where}\n\n{bar}\n\n<sub>{legend}</sub>"

    def _scenario_detail_updates(self, sc):
        """Build the (editor, desc, steps_view, policy, fresh, status) updates
        shown when a scenario is selected or the list is reloaded."""
        if sc is None:
            empty_status = self._scenario_status_md(0, {}, "idle")
            return (
                "",
                "_No scenarios found. Add `.txt` files to the `scenarios/` "
                "folder, then click **↻ Reload**._",
                "_No steps to show._",
                gr.update(),
                gr.update(),
                empty_status,
            )
        desc = (
            f"**{sc.title}**\n\n"
            f"{sc.description or '_No description provided._'}\n\n"
            f"*{sc.num_steps} step(s)*"
        )
        steps_view = self._steps_view_md(sc, -1, ["pending"] * sc.num_steps)
        return (
            sc.raw,
            desc,
            steps_view,
            gr.update(value=scenario_lib.policy_label(sc.safety)),
            gr.update(value=sc.fresh_context),
            self._scenario_status_md(sc.num_steps, {}, "idle"),
        )

    def select_scenario(self, title: Optional[str]):
        """Dropdown change handler: populate editor/preview/options."""
        return self._scenario_detail_updates(self._find_scenario(title))

    def reload_scenarios(self):
        """Re-scan the scenarios directory and refresh the picker + preview."""
        self._scenarios = load_scenarios()
        titles = self._scenario_titles()
        value = titles[0] if titles else None
        sc = self._scenarios[0] if self._scenarios else None
        return (gr.update(choices=titles, value=value),) + self._scenario_detail_updates(sc)

    def pause_scenario(self):
        """Pause the run (takes effect between steps)."""
        self._scenario_pause.set()
        logger.info("Scenario paused")
        return gr.update(visible=False), gr.update(visible=True)

    def resume_scenario(self):
        """Resume a paused run."""
        self._scenario_pause.clear()
        logger.info("Scenario resumed")
        return gr.update(visible=True), gr.update(visible=False)

    def stop_scenario(self):
        """Stop the run and immediately stop the robot, then reset controls."""
        self._scenario_stop.set()
        self._scenario_pause.clear()
        try:
            self.emergency_stop()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Emergency stop during scenario stop failed: {e}")
        logger.info("Scenario stop requested")
        # run, pause, resume, stop -> back to idle layout
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def run_scenario_stream(
        self,
        raw_text: str,
        name_hint: Optional[str],
        policy_label: str,
        max_attempts: Any,
        fresh_context: bool,
        history: Optional[list],
    ) -> Generator[tuple, None, None]:
        """Run a scenario, guarding against a second concurrent run.

        Thin wrapper around :meth:`_run_scenario_core` that sets the
        ``_scenario_active`` re-entrancy flag and always clears it (even on
        cancel/exception) via ``finally``.
        """
        history = list(history or [])
        if self._scenario_active:
            history.append({
                "role": "assistant",
                "content": "⚠️ A scenario is already running. Stop it before starting another.",
            })
            yield history, gr.update(), gr.update()
            return
        self._scenario_active = True
        try:
            yield from self._run_scenario_core(
                raw_text, name_hint, policy_label, max_attempts, fresh_context, history
            )
        finally:
            self._scenario_active = False

    def _run_scenario_core(
        self,
        raw_text: str,
        name_hint: Optional[str],
        policy_label: str,
        max_attempts: Any,
        fresh_context: bool,
        history: Optional[list],
    ) -> Generator[tuple, None, None]:
        """Execute the scenario in ``raw_text`` step by step, streaming the
        transcript, progress bar, and step states into the Scenarios tab.

        Yields ``(chatbot_history, status_md, steps_view_md)`` tuples.
        """
        history = list(history or [])

        # NOTE: control flags (_scenario_stop/_scenario_pause) are cleared by the
        # Run button's first handler (_scenario_running_ui), before the Stop
        # button is shown, so we must NOT clear them here — doing so could race
        # with and discard a Stop press.

        name = re.sub(r"\W+", "_", (name_hint or "custom")).strip("_").lower() or "custom"
        scenario = parse_scenario_text(raw_text or "", name=name)
        policy = LABEL_TO_POLICY.get(policy_label, POLICY_STOP)
        try:
            max_attempts = int(max_attempts)
        except (TypeError, ValueError):
            max_attempts = 2
        max_attempts = max(0, min(max_attempts, 5))

        total = scenario.num_steps
        statuses = ["pending"] * total
        counters = {"ok": 0, "recovered": 0, "failed": 0}
        current = -1

        if total == 0:
            history.append({
                "role": "assistant",
                "content": "⚠️ This scenario has no steps. Add some commands (one per line) and run again.",
            })
            yield history, self._scenario_status_md(0, counters, "idle"), self._steps_view_md(scenario, -1, statuses)
            return

        # Optional fresh conversation context for this run.
        if fresh_context and self.agent is not None:
            try:
                self.agent.clear_history()
            except Exception as e:
                logger.debug(f"clear_history failed (non-fatal): {e}")

        # Safety supervisor reuses the agent's own LLM; None in simple mode.
        llm = None
        if self.agent is not None:
            try:
                llm = self.agent.get_llm()
            except Exception:
                llm = None
        supervisor = SafetySupervisor(llm) if policy == POLICY_SUPERVISE else None

        # All invokes funnel through _invoke_agent (agent lock + None-safe).
        invoke = self._invoke_agent

        def estop():
            try:
                self.emergency_stop()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Emergency stop callback failed: {e}")

        # Opening banner.
        history.append({
            "role": "assistant",
            "content": (
                f"▶ **Scenario:** {scenario.title} — {total} step(s) · "
                f"safety: _{policy_label}_"
            ),
        })
        yield history, self._scenario_status_md(total, counters, "running", current), self._steps_view_md(scenario, current, statuses)

        for ev in run_scenario(
            scenario,
            invoke=invoke,
            emergency_stop=estop,
            supervisor=supervisor,
            policy=policy,
            max_attempts=max_attempts,
            stop_event=self._scenario_stop,
            pause_event=self._scenario_pause,
        ):
            kind = ev.get("kind")

            if kind == "scenario_start":
                continue

            if kind == "paused":
                current = ev.get("index", current)
                yield history, self._scenario_status_md(total, counters, "paused", current), self._steps_view_md(scenario, current, statuses)
                continue

            if kind == "step_start":
                current = ev["index"]
                if 0 <= current < total:
                    statuses[current] = "running"
                prefix = "↻ " if ev.get("attempt", 0) > 0 else ""
                label = f"{prefix}Step {current + 1}/{total}"
                history.append({"role": "user", "content": f"**{label}** · {ev['prompt']}"})
                yield history, self._scenario_status_md(total, counters, "running", current), self._steps_view_md(scenario, current, statuses)
                continue

            if kind == "step_result":
                text = ev.get("output") or "_(no output)_"
                tools = ev.get("tools") or []
                if tools:
                    text = f"{text}\n\n<sub>tools: {', '.join(tools)}</sub>"
                if ev.get("mitigation"):
                    text = f"🩹 _mitigation result_\n\n{text}"
                history.append({"role": "assistant", "content": text})
                yield history, self._scenario_status_md(total, counters, "running", current), self._steps_view_md(scenario, current, statuses)
                continue

            if kind == "supervisor":
                action = ev.get("action", "")
                emoji = {"ok": "✅", "retry": "🔁", "mitigate": "🩹", "abort": "🛑"}.get(action, "🛡️")
                msg = f"🛡️ **Safety supervisor:** {emoji} `{action}` — {ev.get('reason', '')}"
                if ev.get("fix"):
                    msg += f"\n\n> {ev['fix']}"
                history.append({"role": "assistant", "content": msg})
                yield history, self._scenario_status_md(total, counters, "running", current), self._steps_view_md(scenario, current, statuses)
                continue

            if kind == "recovery":
                kl = ev.get("kind_label", "recovery")
                history.append({
                    "role": "user",
                    "content": f"🩹 **Recovery ({kl}):** {ev.get('prompt', '')}",
                })
                yield history, self._scenario_status_md(total, counters, "running", current), self._steps_view_md(scenario, current, statuses)
                continue

            if kind == "info":
                history.append({"role": "assistant", "content": f"ℹ️ {ev.get('text', '')}"})
                yield history, self._scenario_status_md(total, counters, "running", current), self._steps_view_md(scenario, current, statuses)
                continue

            if kind == "step_done":
                counters = ev.get("counters", counters)
                idx = ev.get("index", current)
                if 0 <= idx < total:
                    statuses[idx] = ev.get("outcome", "ok")
                yield history, self._scenario_status_md(total, counters, "running", current), self._steps_view_md(scenario, current, statuses)
                continue

            if kind == "stopped":
                counters = ev.get("counters", counters)
                statuses = ["pending" if s == "running" else s for s in statuses]
                history.append({"role": "assistant", "content": "⏹ **Scenario stopped.** Robot velocity zeroed."})
                yield history, self._scenario_status_md(total, counters, "stopped"), self._steps_view_md(scenario, -1, statuses)
                return

            if kind == "aborted":
                counters = ev.get("counters", counters)
                idx = ev.get("index", current)
                if 0 <= idx < total:
                    statuses[idx] = "failed"
                history.append({
                    "role": "assistant",
                    "content": (
                        f"🛑 **Scenario aborted at step {idx + 1}:** {ev.get('reason', '')}\n\n"
                        "Emergency stop engaged — the robot has been stopped."
                    ),
                })
                yield history, self._scenario_status_md(total, counters, "aborted"), self._steps_view_md(scenario, -1, statuses)
                return

            if kind == "done":
                counters = ev.get("counters", counters)
                summary = (
                    f"✅ **Scenario complete** — {counters.get('ok', 0)} ok, "
                    f"{counters.get('recovered', 0)} recovered, "
                    f"{counters.get('failed', 0)} failed of {total} step(s)."
                )
                history.append({"role": "assistant", "content": summary})
                yield history, self._scenario_status_md(total, counters, "done"), self._steps_view_md(scenario, -1, statuses)
                return

    def create_ui(self) -> gr.Blocks:
        """Create the Gradio UI interface."""

        # Get path to robot image
        assets_dir = Path(__file__).parent / "assets"
        robot_image_path = assets_dir / "robot_ranger_garden.webp"

        # Load pre-made scenarios for the Scenarios tab and compute defaults.
        self._scenarios = load_scenarios()
        _sc_titles = self._scenario_titles()
        _sc_default = self._scenarios[0] if self._scenarios else None
        _sc_default_title = _sc_titles[0] if _sc_titles else None
        if _sc_default is not None:
            _sc_raw = _sc_default.raw
            _sc_desc = (
                f"**{_sc_default.title}**\n\n"
                f"{_sc_default.description or '_No description provided._'}\n\n"
                f"*{_sc_default.num_steps} step(s)*"
            )
            _sc_steps = self._steps_view_md(
                _sc_default, -1, ["pending"] * _sc_default.num_steps
            )
            _sc_policy = scenario_lib.policy_label(_sc_default.safety)
            _sc_fresh = _sc_default.fresh_context
            _sc_status = self._scenario_status_md(_sc_default.num_steps, {}, "idle")
        else:
            _sc_raw = ""
            _sc_desc = (
                "_No scenarios found. Add `.txt` files to the `scenarios/` "
                "folder, then click **↻ Reload**._"
            )
            _sc_steps = "_No steps to show._"
            _sc_policy = POLICY_LABELS[POLICY_STOP]
            _sc_fresh = True
            _sc_status = self._scenario_status_md(0, {}, "idle")

        with gr.Blocks(title="Ranger Robot Assistant") as demo:
            # Centered title
            gr.Markdown(
                """
                <h1 style="text-align: center;">Ranger Robot Assistant</h1>
                """,
                elem_id="title"
            )

            # Create three tabs: Home, Status, Settings (centered)
            with gr.Tabs(elem_classes=["tabs"]):
                # Home Tab - Main chat interface with robot image
                with gr.Tab("Home"):
                    # Robot image at the top (larger)
                    with gr.Row():
                        with gr.Column(scale=1):
                            pass
                        with gr.Column(scale=2):
                            if robot_image_path.exists():
                                gr.Image(
                                    value=str(robot_image_path),
                                    label=None,
                                    show_label=False,
                                    container=False,
                                    height=280,
                                    buttons=[],  # Hide all buttons (download, share, fullscreen)
                                )
                        with gr.Column(scale=1):
                            pass

                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=500,
                    )

                    # Input row: text + inline mic (icon only) + Send, like a chat app
                    with gr.Row(elem_classes=["input-row"]):
                        msg = gr.Textbox(
                            label="Command",
                            placeholder="Type a command like 'move forward 1 meter'",
                            show_label=False,
                            container=False,
                            scale=8,
                        )
                        # Mic: record -> speech-to-text -> command box (auto-sends).
                        # Styled down to a single mic button via the .mic-compact CSS.
                        mic_in = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            show_label=False,
                            container=False,
                            scale=0,
                            min_width=56,
                            elem_classes=["mic-compact"],
                        )
                        submit_btn = gr.Button(
                            "Send", variant="primary", scale=0, min_width=90
                        )
                        stop_chat_btn = gr.Button(
                            "Stop", variant="stop", scale=0, min_width=80, visible=False
                        )

                    # Text-to-speech output for the assistant reply. Kept rendered
                    # (autoplay needs it in the DOM) but shrunk via .tts-mini CSS.
                    tts_audio = gr.Audio(
                        label="Voice reply",
                        show_label=False,
                        autoplay=True,
                        interactive=False,
                        elem_classes=["tts-mini"],
                    )

                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                        speak_toggle = gr.Checkbox(
                            value=True,
                            label="🔊 Speak replies",
                            scale=0,
                            min_width=140,
                        )
                        stop_btn = gr.Button(
                            "EMERGENCY STOP",
                            variant="stop",
                            elem_classes=["stop-button"],
                        )

                # Scenarios Tab - Pick & run a pre-made scenario (prompt file
                # fed line-by-line) with a live transcript and a safety net.
                with gr.Tab("Scenarios"):
                    gr.Markdown(
                        "### Pre-made Scenarios\n"
                        "Pick a scenario, review the steps, then **▶ Run** it. "
                        "Each line is sent to me in order (context carries over), "
                        "with a safety net that stops — or recovers — when a step "
                        "goes wrong. Edit the steps under **✎ Edit scenario** to "
                        "customize a run."
                    )
                    with gr.Row():
                        # Left column: picker + options + controls
                        with gr.Column(scale=1):
                            with gr.Row(elem_classes=["input-row"]):
                                scenario_dropdown = gr.Dropdown(
                                    choices=_sc_titles,
                                    value=_sc_default_title,
                                    label="Scenario",
                                    interactive=True,
                                    scale=4,
                                )
                                reload_scenarios_btn = gr.Button(
                                    "↻", scale=0, min_width=44, size="sm"
                                )
                            scenario_desc = gr.Markdown(_sc_desc)

                            with gr.Accordion("Safety & options", open=False):
                                scenario_policy = gr.Radio(
                                    choices=list(POLICY_LABELS.values()),
                                    value=_sc_policy,
                                    label="On step error",
                                    info=(
                                        "Stop = halt + e-stop · AI supervisor = "
                                        "let me adjudicate & recover · Run all = "
                                        "ignore (manual e-stop only)"
                                    ),
                                )
                                scenario_max_attempts = gr.Number(
                                    value=2,
                                    precision=0,
                                    minimum=0,
                                    maximum=5,
                                    label="Max recovery attempts (AI supervisor)",
                                )
                                scenario_fresh = gr.Checkbox(
                                    value=_sc_fresh,
                                    label="Start from fresh context (clear chat history)",
                                )

                            with gr.Accordion("✎ Edit scenario (advanced)", open=False):
                                scenario_editor = gr.Textbox(
                                    value=_sc_raw,
                                    label="Scenario steps — one command per line ('#' = comment)",
                                    lines=10,
                                    max_lines=20,
                                    interactive=True,
                                )

                            with gr.Row():
                                run_scenario_btn = gr.Button(
                                    "▶ Run", variant="primary", scale=2
                                )
                                pause_scenario_btn = gr.Button(
                                    "⏸ Pause", scale=1, visible=False
                                )
                                resume_scenario_btn = gr.Button(
                                    "▶ Resume", scale=1, visible=False
                                )
                                scenario_stop_btn = gr.Button(
                                    "⏹ Stop", variant="stop", scale=1, visible=False
                                )
                            scenario_estop_btn = gr.Button(
                                "🛑 EMERGENCY STOP",
                                variant="stop",
                                elem_classes=["stop-button"],
                            )

                        # Right column: progress + steps + live transcript
                        with gr.Column(scale=2):
                            scenario_status = gr.Markdown(_sc_status)
                            with gr.Accordion("Steps", open=True):
                                scenario_steps_view = gr.Markdown(_sc_steps)
                            scenario_chatbot = gr.Chatbot(
                                label="Scenario run",
                                height=420,
                            )

                # Status Tab - Status display and controls
                with gr.Tab("Status"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Battery Status")
                            battery_display = gr.Textbox(
                                label="Battery",
                                value=self.get_battery_status(),
                                interactive=False,
                            )
                            refresh_btn = gr.Button("Refresh Status", size="sm")
                            battery_timer = gr.Timer(value=5)

                        with gr.Column(scale=1):
                            gr.Markdown("### Camera")
                            camera_image = gr.Image(
                                label="Camera",
                                value=self.get_camera_image(),
                                interactive=False,
                                height=240,
                            )
                            camera_refresh_btn = gr.Button("Refresh Camera", size="sm")

                    gr.Markdown("### Manual Controls")
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Empty space on the left
                            pass
                        with gr.Column(scale=1):
                            with gr.Row():
                                gr.Button("").visible = False  # Spacer
                                fwd_btn = gr.Button("↑ Forward", scale=1)
                                gr.Button("").visible = False  # Spacer

                            with gr.Row():
                                left_btn = gr.Button("← Left", scale=1)
                                stop_manual_btn = gr.Button("Stop", scale=1)
                                right_btn = gr.Button("Right →", scale=1)

                            with gr.Row():
                                gr.Button("").visible = False  # Spacer
                                back_btn = gr.Button("↓ Back", scale=1)
                                gr.Button("").visible = False  # Spacer

                            manual_output = gr.Textbox(
                                label="Manual Control Output",
                                interactive=False,
                                lines=2,
                            )
                        with gr.Column(scale=1):
                            # Empty space on the right
                            pass

                # Settings Tab - Gradio and system settings
                with gr.Tab("Settings"):
                    gr.Markdown("### System Information")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(f"""
                            - **LLM Provider:** {self.llm_provider}
                            - **Model:** {self.model_name or 'Default'}
                            - **Simple Mode:** {'Yes' if self.simple_mode else 'No'}
                            - **ROS 2 Available:** {'Yes' if ROS_AVAILABLE else 'No'}
                            - **Server Port:** {self.server_port}
                            """)

                    gr.Markdown("### Model Selection")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(
                                "Choose Claude model. Applies on selection (re-initializes agent). "
                                "Only relevant when provider is `claude_code` or `anthropic`."
                            )
                            model_choices = [
                                "sonnet-4.6",
                                "opus-4.8",
                                "opus-4.7",
                                "haiku-4.5",
                                "sonnet-4",
                            ]
                            default_model = self.model_name if self.model_name in model_choices else "sonnet-4.6"
                            model_dropdown = gr.Dropdown(
                                choices=model_choices,
                                value=default_model,
                                label="Claude Model",
                                interactive=True,
                                allow_custom_value=True,
                            )
                            model_status = gr.Markdown("")

                    gr.Markdown("### Agent Mode")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(
                                "**Running mode** uses the full Ranger persona + ROSA base prompts.  \n"
                                "**Debug mode** sends only the ROSA base system prompts — no Ranger-specific instructions."
                            )
                            debug_toggle = gr.Radio(
                                choices=["Running", "Debug"],
                                value="Debug" if self.debug_mode else "Running",
                                label="Agent Mode",
                                interactive=True,
                            )
                            mode_status = gr.Markdown("")

                    gr.Markdown("### Voice (Speech-to-Text / Text-to-Speech)")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(
                                "Local, offline voice via **faster-whisper** (STT) "
                                "and **Piper** (TTS). No cloud API key needed. "
                                "Record with the mic on the Home tab; toggle "
                                "**Speak replies** to hear responses.\n\n"
                                "**Environment Variables:**\n"
                                "- `WHISPER_MODEL`: STT model (default: small.en; also base.en, medium.en)\n"
                                "- `WHISPER_DEVICE`: auto, cuda, or cpu (default: auto)\n"
                                "- `PIPER_VOICE`: TTS voice (default: en_US-lessac-medium)\n"
                                "- `PIPER_VOICE_PATH`: path to a local .onnx voice (skips download)\n"
                                "- `PIPER_DOWNLOAD`: allow voice auto-download (default: enabled)"
                            )
                            voice_status_box = gr.Textbox(
                                label="Voice Backend Status",
                                value="Click 'Check Voice Backends' to load models",
                                interactive=False,
                                lines=2,
                            )
                            voice_status_btn = gr.Button(
                                "Check Voice Backends", size="sm"
                            )

                    gr.Markdown("### Gradio Settings")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("""
                            **Current Configuration:**
                            - Server: 0.0.0.0
                            - Show Error: Enabled

                            **Environment Variables:**
                            You can configure the following via environment variables:
                            - `LLM_PROVIDER`: openai, ollama, anthropic, or claude_code
                            - `LLM_MODEL`: Model name (claude_code defaults: sonnet-4.6; also opus-4.8, opus-4.7, haiku-4.5, sonnet-4)
                            - `CLAUDE_CODE_OAUTH_TOKEN`: OAuth token for Claude Pro/Max subscription
                            - `GRADIO_PORT`: Server port (default: 7860)
                            - `SHOW_LLM_USAGE`: Show token usage (true/false)

                            **Camera Settings:**
                            - `CAMERA_IMAGE_MAX_WIDTH`: Image width (default: 320)
                            - `CAMERA_IMAGE_MAX_HEIGHT`: Image height (default: 240)
                            - `CAMERA_IMAGE_QUALITY`: JPEG quality (default: 75)
                            - `CAMERA_IMAGE_FORMAT`: jpeg or png (default: jpeg)
                            - Named views (ask "show the wrist cam" / "rear camera"):
                              `front` (default), `wrist`, `rear` (fixed D435i behind the arm).
                              `CAMERA_DEFAULT`, `CAMERA_WRIST_TOPIC`, `CAMERA_REAR_SERIAL`
                            """)

            # Event handler for model selection
            model_dropdown.change(
                fn=self.set_model,
                inputs=[model_dropdown],
                outputs=[model_status],
            )

            # Event handler for voice backend status check
            voice_status_btn.click(
                fn=voice_status,
                outputs=[voice_status_box],
            )

            # Event handler for debug/running mode toggle
            debug_toggle.change(
                fn=lambda choice: self.set_debug_mode(choice == "Debug"),
                inputs=[debug_toggle],
                outputs=[mode_status],
            )

            # Event handlers for Home tab
            # When Send is clicked: hide Send, show Stop, run chat
            submit_event = submit_btn.click(
                fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                inputs=None,
                outputs=[submit_btn, stop_chat_btn],
            ).then(
                fn=self.chat_response,
                inputs=[msg, chatbot],
                outputs=[chatbot],
            ).then(
                fn=lambda: "",
                outputs=[msg],
            ).then(
                fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                inputs=None,
                outputs=[submit_btn, stop_chat_btn],
            ).then(
                fn=self.synthesize_response,
                inputs=[chatbot, speak_toggle],
                outputs=[tts_audio],
            )

            # When Enter is pressed: hide Send, show Stop, run chat
            msg_event = msg.submit(
                fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                inputs=None,
                outputs=[submit_btn, stop_chat_btn],
            ).then(
                fn=self.chat_response,
                inputs=[msg, chatbot],
                outputs=[chatbot],
            ).then(
                fn=lambda: "",
                outputs=[msg],
            ).then(
                fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                inputs=None,
                outputs=[submit_btn, stop_chat_btn],
            ).then(
                fn=self.synthesize_response,
                inputs=[chatbot, speak_toggle],
                outputs=[tts_audio],
            )

            # When mic recording stops: transcribe -> fill command box -> send.
            # Also reset the mic component to None so it snaps back to the plain
            # mic button instead of Gradio's post-record waveform/player editor
            # (ChatGPT-style: tap mic, talk, it transcribes & sends, done).
            mic_event = mic_in.stop_recording(
                fn=lambda p: (self.transcribe_audio(p), None),
                inputs=[mic_in],
                outputs=[msg, mic_in],
            ).then(
                fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                inputs=None,
                outputs=[submit_btn, stop_chat_btn],
            ).then(
                fn=self.chat_response,
                inputs=[msg, chatbot],
                outputs=[chatbot],
            ).then(
                fn=lambda: "",
                outputs=[msg],
            ).then(
                fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                inputs=None,
                outputs=[submit_btn, stop_chat_btn],
            ).then(
                fn=self.synthesize_response,
                inputs=[chatbot, speak_toggle],
                outputs=[tts_audio],
            )

            # ---- Scenarios tab wiring ----
            # Selecting a scenario populates the editor, preview, and options.
            scenario_dropdown.change(
                fn=self.select_scenario,
                inputs=[scenario_dropdown],
                outputs=[
                    scenario_editor,
                    scenario_desc,
                    scenario_steps_view,
                    scenario_policy,
                    scenario_fresh,
                    scenario_status,
                ],
            )

            # Reload re-scans the scenarios/ folder and refreshes everything.
            reload_scenarios_btn.click(
                fn=self.reload_scenarios,
                inputs=None,
                outputs=[
                    scenario_dropdown,
                    scenario_editor,
                    scenario_desc,
                    scenario_steps_view,
                    scenario_policy,
                    scenario_fresh,
                    scenario_status,
                ],
            )

            # Run: swap to the running control layout, stream the scenario, then
            # restore the idle layout when it completes (normally or via abort).
            # Clearing the control flags here (before the Stop button is shown)
            # — rather than inside the streaming generator — means a Stop press
            # can never be clobbered by a late clear.
            def _scenario_running_ui():
                self._scenario_stop.clear()
                self._scenario_pause.clear()
                return (
                    gr.update(visible=False),  # run
                    gr.update(visible=True),   # pause
                    gr.update(visible=False),  # resume
                    gr.update(visible=True),   # stop
                )

            def _scenario_idle_ui():
                return (
                    gr.update(visible=True),   # run
                    gr.update(visible=False),  # pause
                    gr.update(visible=False),  # resume
                    gr.update(visible=False),  # stop
                )

            scenario_run_event = run_scenario_btn.click(
                fn=_scenario_running_ui,
                inputs=None,
                outputs=[
                    run_scenario_btn,
                    pause_scenario_btn,
                    resume_scenario_btn,
                    scenario_stop_btn,
                ],
            ).then(
                fn=self.run_scenario_stream,
                inputs=[
                    scenario_editor,
                    scenario_dropdown,
                    scenario_policy,
                    scenario_max_attempts,
                    scenario_fresh,
                    scenario_chatbot,
                ],
                outputs=[scenario_chatbot, scenario_status, scenario_steps_view],
            ).then(
                fn=_scenario_idle_ui,
                inputs=None,
                outputs=[
                    run_scenario_btn,
                    pause_scenario_btn,
                    resume_scenario_btn,
                    scenario_stop_btn,
                ],
            )

            # Pause/Resume swap their two buttons; the runner honors the flag
            # between steps.
            pause_scenario_btn.click(
                fn=self.pause_scenario,
                inputs=None,
                outputs=[pause_scenario_btn, resume_scenario_btn],
            )
            resume_scenario_btn.click(
                fn=self.resume_scenario,
                inputs=None,
                outputs=[pause_scenario_btn, resume_scenario_btn],
            )

            # Stop / E-stop: halt the run, stop the robot, cancel the generator,
            # and reset the controls to idle.
            scenario_stop_btn.click(
                fn=self.stop_scenario,
                inputs=None,
                outputs=[
                    run_scenario_btn,
                    pause_scenario_btn,
                    resume_scenario_btn,
                    scenario_stop_btn,
                ],
                cancels=[scenario_run_event],
            )
            scenario_estop_btn.click(
                fn=self.stop_scenario,
                inputs=None,
                outputs=[
                    run_scenario_btn,
                    pause_scenario_btn,
                    resume_scenario_btn,
                    scenario_stop_btn,
                ],
                cancels=[scenario_run_event],
            )

            # When Stop is clicked: cancel chat, restore Send button
            stop_chat_btn.click(
                fn=self.cancel_chat,
                inputs=None,
                outputs=[submit_btn, stop_chat_btn],
                cancels=[submit_event, msg_event, mic_event],
            )

            clear_btn.click(
                fn=lambda: [],
                outputs=[chatbot],
            )

            # Stop button cancels the ongoing chat request AND any running
            # scenario, then executes emergency stop. The trailing .then resets
            # the scenario controls to idle — the run chain's own
            # .then(_scenario_idle_ui) won't fire because it was just cancelled.
            stop_btn.click(
                fn=self.emergency_stop,
                outputs=[],
                cancels=[submit_event, msg_event, mic_event, scenario_run_event],
            ).then(
                fn=_scenario_idle_ui,
                inputs=None,
                outputs=[
                    run_scenario_btn,
                    pause_scenario_btn,
                    resume_scenario_btn,
                    scenario_stop_btn,
                ],
            )

            # Event handlers for Status tab
            refresh_btn.click(
                fn=self.get_battery_status,
                outputs=[battery_display],
            )
            battery_timer.tick(
                fn=self.get_battery_status,
                outputs=[battery_display],
            )

            camera_refresh_btn.click(
                fn=self.get_camera_image,
                outputs=[camera_image],
            )

            # Manual control handlers
            fwd_btn.click(fn=self.teleop_forward, outputs=[manual_output])
            back_btn.click(fn=self.teleop_backward, outputs=[manual_output])
            left_btn.click(fn=self.teleop_left, outputs=[manual_output])
            right_btn.click(fn=self.teleop_right, outputs=[manual_output])
            # Manual Stop also halts a running scenario (parity with the other
            # E-stops): cancel the run chain and reset the scenario controls.
            stop_manual_btn.click(
                fn=self.emergency_stop,
                outputs=[manual_output],
                cancels=[scenario_run_event],
            ).then(
                fn=_scenario_idle_ui,
                inputs=None,
                outputs=[
                    run_scenario_btn,
                    pause_scenario_btn,
                    resume_scenario_btn,
                    scenario_stop_btn,
                ],
            )

            # Custom footer at the bottom
            gr.HTML(
                """
                <div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid #e0e0e0;">
                    <a href="https://github.com/anh0001/ranger-garden-assistant" target="_blank" style="color: #0066cc; text-decoration: none; font-size: 14px;">
                        GitHub: Ranger Robot Assistant
                    </a>
                </div>
                """,
                elem_id="custom-footer"
            )

        return demo

    def run(self):
        """Run the UI node."""
        logger.info("Starting Ranger LLM UI...")

        # Initialize ROS 2
        self.initialize_ros()

        # Initialize agent
        self.initialize_agent()

        # Create and launch UI
        demo = self.create_ui()

        try:
            # Custom CSS to center tabs and hide Gradio footer
            custom_css = """
                /* Center the tab navigation buttons only */
                div[role="tablist"] {
                    justify-content: center !important;
                }

                /* Hide Gradio footer */
                footer {
                    display: none !important;
                }

                /* Keep the chat input row items on one line, vertically centered */
                .input-row {
                    align-items: center !important;
                    gap: 6px !important;
                }

                /* Compact mic: strip the box and all the recorder chrome, leaving
                   one round button that shows a mic glyph (like a chat app). */
                .mic-compact {
                    flex: 0 0 auto !important;
                    min-width: 48px !important;
                    max-width: 48px !important;
                    border: none !important;
                    background: transparent !important;
                    box-shadow: none !important;
                    padding: 0 !important;
                    overflow: visible !important;
                }
                /* Kill the device dropdown ("No microphone found"), the clear/X
                   icon button, the empty waveform canvas, and any leftover label
                   (Gradio 6.x DOM: .mic-select, .icon-button-wrapper, the
                   .microphone / recording-waveform canvases). */
                .mic-compact .mic-select,
                .mic-compact .icon-button-wrapper,
                .mic-compact .microphone,
                .mic-compact [data-testid="microphone-waveform"],
                .mic-compact [data-testid="recording-waveform"],
                .mic-compact label,
                .mic-compact .label-icon {
                    display: none !important;
                }
                /* Collapse the nested audio containers so the lone button sits
                   inline with no surrounding box or vertical padding. */
                .mic-compact .audio-container,
                .mic-compact .component-wrapper,
                .mic-compact .controls,
                .mic-compact .controls .wrapper {
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    gap: 0 !important;
                    min-height: 0 !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    border: none !important;
                    background: transparent !important;
                    box-shadow: none !important;
                }
                /* Round buttons; hide the injected "Record"/"Stop" text via
                   font-size:0 and draw a glyph with ::before instead. In Gradio
                   6.x there is NO .record-icon element — the glyph must live on
                   the button itself.

                   IMPORTANT: do NOT set `display` here. Gradio keeps all five
                   recorder buttons in the DOM and toggles them by recording
                   state with `display:none`; overriding display would un-hide
                   them all and stack them vertically. Center the glyph with
                   line-height (vertical) + text-align (horizontal) instead. */
                .mic-compact .record-button,
                .mic-compact .stop-button,
                .mic-compact .stop-button-paused,
                .mic-compact .pause-button,
                .mic-compact .resume-button {
                    font-size: 0 !important;
                    gap: 0 !important;
                    width: 44px !important;
                    min-width: 44px !important;
                    height: 44px !important;
                    padding: 0 !important;
                    border-radius: 50% !important;
                    text-align: center !important;
                }
                /* Gradio's own .record-button:before draws a small orange record
                   dot (content:""; background:var(--primary-600); fixed 16px;
                   border-radius:full). Fully neutralize it — drop the background,
                   the fixed size and the margin — so only our 🎤 glyph renders. */
                .mic-compact .record-button::before {
                    content: "🎤" !important;
                    background: none !important;
                    width: auto !important;
                    height: auto !important;
                    margin: 0 !important;
                    border-radius: 0 !important;
                    font-size: 20px !important;
                    line-height: 44px !important;
                }
                /* While recording, the record button is swapped for a red stop
                   button — give it a square "stop" glyph so it reads clearly. */
                .mic-compact .stop-button::before,
                .mic-compact .stop-button-paused::before {
                    content: "⏹" !important;
                    background: none !important;
                    width: auto !important;
                    height: auto !important;
                    margin: 0 !important;
                    border-radius: 0 !important;
                    font-size: 20px !important;
                    line-height: 44px !important;
                }
                /* After a recording stops, Gradio briefly flips gr.Audio into a
                   waveform PLAYER/EDITOR (play, rewind, skip, speed, volume,
                   trim, undo) that overflows and overlaps the Send button. We
                   reset the value to None on stop (see stop_recording handler)
                   so it snaps back to the mic button, but hide all the player
                   chrome too so the transcription window never shows it. */
                .mic-compact .waveform-container,
                .mic-compact .timestamps,
                .mic-compact .control-wrapper,
                .mic-compact .play-pause-wrapper,
                .mic-compact .settings-wrapper,
                .mic-compact .playback,
                .mic-compact .play-pause-button,
                .mic-compact .rewind,
                .mic-compact .skip,
                .mic-compact .volume,
                .mic-compact .volume-control-wrapper,
                .mic-compact .standard-player,
                .mic-compact .timeline-wrapper,
                .mic-compact .action-buttons,
                .mic-compact button.action {
                    display: none !important;
                }

                /* Voice-reply player: keep it rendered (needed for autoplay) but
                   unobtrusive — a thin strip, no big container. */
                .tts-mini {
                    border: none !important;
                    background: transparent !important;
                    box-shadow: none !important;
                    margin-top: 4px !important;
                }
                .tts-mini .controls {
                    min-height: 0 !important;
                }
            """

            # JavaScript to force light theme - using IIFE (Immediately Invoked Function Expression)
            js_func = """
            (function() {
                const url = new URL(window.location);
                if (url.searchParams.get('__theme') !== 'light') {
                    url.searchParams.set('__theme', 'light');
                    window.location.href = url.href;
                }
            })();
            """

            # Configure queue with proper concurrency settings
            demo.queue(
                default_concurrency_limit=1,  # Process 1 request at a time (LLM is resource-heavy)
                max_size=5,  # Limit queue to 5 requests to prevent long waits
                api_open=True,  # Enable API access
            )

            # Bind address is configurable. For the Tailscale Serve HTTPS path
            # (needed so the browser mic works over the tailnet), set
            # GRADIO_SERVER_NAME=127.0.0.1 and front it with `tailscale serve`.
            server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
            demo.launch(
                server_name=server_name,
                server_port=self.server_port,
                share=self.share,
                show_error=True,
                css=custom_css,
                js=js_func,
            )
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down Ranger LLM UI...")

        if self._node is not None and ROS_AVAILABLE:
            try:
                # Only shutdown if context is still valid
                if rclpy.ok():
                    self._node.destroy_node()
                    rclpy.shutdown()
            except Exception as e:
                logger.warning(f"Error during shutdown: {e}")

        logger.info("Shutdown complete")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ranger LLM UI")
    parser.add_argument(
        "--provider",
        type=str,
        default=os.getenv("LLM_PROVIDER", "openai"),
        choices=["openai", "ollama", "anthropic", "claude_code", "claude_proxy"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LLM_MODEL"),
        help="Model name (default: provider default)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GRADIO_PORT", "7860")),
        help="Gradio server port (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple agent without LLM (for testing)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.getenv("RANGER_DEBUG_MODE", "").lower() in {"1", "true", "yes"},
        help="Debug mode: use only ROSA base prompts, skip Ranger-specific prompts (default: false)",
    )

    args = parser.parse_args()

    # Convert empty model string to None (use provider default)
    model_name = args.model if args.model else None

    node = RangerUINode(
        llm_provider=args.provider,
        model_name=model_name,
        server_port=args.port,
        share=args.share,
        simple_mode=args.simple,
        debug_mode=args.debug,
    )
    node.run()


if __name__ == "__main__":
    main()
