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
        """
        self.node_name = node_name
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.server_port = server_port
        self.share = share
        self.simple_mode = simple_mode

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

            # Start spinning in background thread
            self._executor = MultiThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
            self._spin_thread.start()

        except Exception as e:
            logger.error(f"Failed to initialize ROS 2: {e}")
            self._node = None

    def _spin_ros(self):
        """Spin ROS 2 node in background."""
        try:
            self._executor.spin()
        except Exception as e:
            logger.error(f"ROS spin error: {e}")

    def initialize_agent(self):
        """Initialize the LangChain agent."""
        try:
            self.agent = create_agent(
                provider=self.llm_provider,
                model_name=self.model_name,
                ros_node=self._node,
                simple_mode=self.simple_mode,
            )
            logger.info(f"Agent initialized with provider: {self.llm_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            # Fall back to simple mode
            logger.info("Falling back to simple mode")
            self.agent = create_agent(ros_node=self._node, simple_mode=True)

    def emergency_stop(self) -> str:
        """Execute emergency stop."""
        ros = get_ros_interface()
        ros.stop()
        logger.warning("EMERGENCY STOP executed")
        return "EMERGENCY STOP executed - Robot stopped"

    def get_battery_status(self) -> str:
        """Get current battery status for display."""
        interface = get_status_interface()
        level, status = interface.get_battery_level()
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
                        output = result.get("output", "I couldn't process that request.")
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

        with gr.Blocks(title="Ranger Garden Assistant") as demo:
            # Centered title
            gr.Markdown(
                """
                <h1 style="text-align: center;">Ranger Garden Assistant</h1>
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

                    with gr.Row():
                        msg = gr.Textbox(
                            label="Command",
                            placeholder="Type a command like 'move forward 1 meter'",
                            scale=4,
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                        stop_chat_btn = gr.Button("Stop", variant="stop", scale=1, visible=False)

                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
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

                    gr.Markdown("### Gradio Settings")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("""
                            **Current Configuration:**
                            - Server: 0.0.0.0
                            - Show Error: Enabled

                            **Environment Variables:**
                            You can configure the following via environment variables:
                            - `LLM_PROVIDER`: openai, ollama, or anthropic
                            - `LLM_MODEL`: Model name
                            - `GRADIO_PORT`: Server port (default: 7860)
                            - `SHOW_LLM_USAGE`: Show token usage (true/false)

                            **Camera Settings:**
                            - `CAMERA_IMAGE_MAX_WIDTH`: Image width (default: 320)
                            - `CAMERA_IMAGE_MAX_HEIGHT`: Image height (default: 240)
                            - `CAMERA_IMAGE_QUALITY`: JPEG quality (default: 75)
                            - `CAMERA_IMAGE_FORMAT`: jpeg or png (default: jpeg)
                            """)

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
            )

            # When Stop is clicked: cancel chat, restore Send button
            stop_chat_btn.click(
                fn=self.cancel_chat,
                inputs=None,
                outputs=[submit_btn, stop_chat_btn],
                cancels=[submit_event, msg_event],
            )

            clear_btn.click(
                fn=lambda: [],
                outputs=[chatbot],
            )

            # Stop button cancels the ongoing chat request AND executes emergency stop
            stop_btn.click(
                fn=self.emergency_stop,
                outputs=[],
                cancels=[submit_event, msg_event],  # Cancel ongoing chat
            )

            # Event handlers for Status tab
            refresh_btn.click(
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
            # Custom CSS to center tabs without breaking functionality
            custom_css = """
                /* Center the tab navigation buttons only */
                div[role="tablist"] {
                    justify-content: center !important;
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

            demo.launch(
                server_name="0.0.0.0",
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
        choices=["openai", "ollama", "anthropic"],
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

    args = parser.parse_args()

    # Convert empty model string to None (use provider default)
    model_name = args.model if args.model else None

    node = RangerUINode(
        llm_provider=args.provider,
        model_name=model_name,
        server_port=args.port,
        share=args.share,
        simple_mode=args.simple,
    )
    node.run()


if __name__ == "__main__":
    main()
