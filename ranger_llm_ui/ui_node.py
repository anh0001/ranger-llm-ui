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
        """Execute emergency stop."""
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
                future = executor.submit(self.agent.invoke, message)

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
        """Flatten chat content to plain text.

        claude_proxy / Anthropic responses may arrive as a list of content
        blocks (e.g. [{"type": "text", "text": "..."}]) rather than a plain
        string, so coerce any shape into speakable text.
        """
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
            result = self.agent.invoke(f"move forward {distance} meters")
            return result.get("output", "Command sent")
        return "Agent not initialized"

    def teleop_backward(self, distance: float = 0.5) -> str:
        """Manual backward movement."""
        if self.agent:
            result = self.agent.invoke(f"move backward {distance} meters")
            return result.get("output", "Command sent")
        return "Agent not initialized"

    def teleop_left(self, angle: float = 45) -> str:
        """Manual left turn."""
        if self.agent:
            result = self.agent.invoke(f"turn left {angle} degrees")
            return result.get("output", "Command sent")
        return "Agent not initialized"

    def teleop_right(self, angle: float = 45) -> str:
        """Manual right turn."""
        if self.agent:
            result = self.agent.invoke(f"turn right {angle} degrees")
            return result.get("output", "Command sent")
        return "Agent not initialized"

    def create_ui(self) -> gr.Blocks:
        """Create the Gradio UI interface."""

        # Get path to robot image
        assets_dir = Path(__file__).parent / "assets"
        robot_image_path = assets_dir / "robot_ranger_garden.webp"

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
                            - `LLM_MODEL`: Model name (claude_code defaults: sonnet-4.6; also opus-4.7, haiku-4.5, sonnet-4)
                            - `CLAUDE_CODE_OAUTH_TOKEN`: OAuth token for Claude Pro/Max subscription
                            - `GRADIO_PORT`: Server port (default: 7860)
                            - `SHOW_LLM_USAGE`: Show token usage (true/false)

                            **Camera Settings:**
                            - `CAMERA_IMAGE_MAX_WIDTH`: Image width (default: 320)
                            - `CAMERA_IMAGE_MAX_HEIGHT`: Image height (default: 240)
                            - `CAMERA_IMAGE_QUALITY`: JPEG quality (default: 75)
                            - `CAMERA_IMAGE_FORMAT`: jpeg or png (default: jpeg)
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

            # Stop button cancels the ongoing chat request AND executes emergency stop
            stop_btn.click(
                fn=self.emergency_stop,
                outputs=[],
                cancels=[submit_event, msg_event, mic_event],  # Cancel ongoing chat
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
            stop_manual_btn.click(fn=self.emergency_stop, outputs=[manual_output])

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
