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
from typing import Optional, Generator, Any
from pathlib import Path
import socket

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


# Get the path to assets directory
ASSETS_DIR = Path(__file__).parent / "assets"
ROBOT_AVATAR_PATH = ASSETS_DIR / "robot_ranger_garden.webp"


def create_ranger_theme() -> gr.themes.Soft:
    """Create custom Ranger dashboard theme with professional styling."""
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        spacing_size="md",
        radius_size="lg",
        text_size="md",
    ).set(
        # Primary button styling
        button_primary_background_fill="#4F46E5",
        button_primary_background_fill_hover="#4338CA",
        button_primary_text_color="white",
        button_primary_border_color="#4F46E5",
        button_primary_shadow="0 4px 6px -1px rgba(79, 70, 229, 0.3)",
        # Secondary button styling
        button_secondary_background_fill="#F1F5F9",
        button_secondary_background_fill_hover="#E2E8F0",
        button_secondary_text_color="#334155",
        # Stop button styling
        button_cancel_background_fill="#EF4444",
        button_cancel_background_fill_hover="#DC2626",
        button_cancel_text_color="white",
        button_cancel_border_color="#EF4444",
        # Block/card styling
        block_background_fill="white",
        block_border_color="#E2E8F0",
        block_shadow="0 1px 3px 0 rgba(0, 0, 0, 0.1)",
        block_title_text_weight="600",
        block_label_text_weight="500",
        # Input styling
        input_background_fill="white",
        input_border_color="#CBD5E1",
        input_border_color_focus="#4F46E5",
        # Body/container
        body_background_fill="#F8FAFC",
    )


# Comprehensive CSS for professional dashboard styling
RANGER_CSS = """
/* ================================================
   RANGER ROBOT CONTROL DASHBOARD - PROFESSIONAL THEME
   ================================================ */

/* === HIDE GRADIO FOOTER === */
footer { display: none !important; }

/* === MAIN CONTAINER === */
.gradio-container {
    max-width: 1600px !important;
    margin: 0 auto !important;
}

/* === HEADER BAR === */
.ranger-header {
    background: linear-gradient(135deg, #4F46E5, #6366F1) !important;
    border-radius: 12px !important;
    padding: 16px 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3) !important;
}

.ranger-header-content {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    flex-wrap: wrap !important;
    gap: 12px !important;
}

.ranger-header-left {
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
}

.ranger-logo {
    width: 48px !important;
    height: 48px !important;
    border-radius: 12px !important;
    object-fit: cover !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
}

.ranger-title {
    color: white !important;
    margin: 0 !important;
}

.ranger-title h1 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    color: white !important;
}

.ranger-title p {
    font-size: 0.875rem !important;
    opacity: 0.9 !important;
    margin: 4px 0 0 0 !important;
    color: white !important;
}

/* === STATUS BADGES === */
.status-badge {
    display: inline-flex !important;
    align-items: center !important;
    padding: 6px 14px !important;
    border-radius: 9999px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    gap: 6px !important;
}

.status-badge::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.status-online {
    background: rgba(16, 185, 129, 0.2) !important;
    color: #D1FAE5 !important;
}

.status-online::before {
    background: #10B981;
    animation: pulse-dot 2s infinite;
}

.status-simulation {
    background: rgba(245, 158, 11, 0.2) !important;
    color: #FEF3C7 !important;
}

.status-simulation::before {
    background: #F59E0B;
}

.status-offline {
    background: rgba(239, 68, 68, 0.2) !important;
    color: #FEE2E2 !important;
}

.status-offline::before {
    background: #EF4444;
}

/* === CARD SECTIONS === */
.ranger-card {
    background: white !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1) !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease !important;
}

.ranger-card:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}

.ranger-card-title {
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid #E2E8F0 !important;
}

/* === BATTERY INDICATOR === */
.battery-container {
    padding: 8px 0 !important;
}

.battery-bar-wrapper {
    width: 100% !important;
    height: 24px !important;
    background: #E2E8F0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    position: relative !important;
}

.battery-bar-fill {
    height: 100% !important;
    border-radius: 12px !important;
    transition: width 0.5s ease, background 0.5s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    padding-right: 8px !important;
    min-width: 40px !important;
}

.battery-bar-fill.high {
    background: linear-gradient(90deg, #10B981, #34D399) !important;
}

.battery-bar-fill.medium {
    background: linear-gradient(90deg, #F59E0B, #FBBF24) !important;
}

.battery-bar-fill.low {
    background: linear-gradient(90deg, #EF4444, #F87171) !important;
}

.battery-text {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    color: white !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
}

.battery-status {
    font-size: 0.75rem !important;
    color: #64748B !important;
    text-align: center !important;
    margin-top: 6px !important;
}

/* === CONTROL PAD === */
.control-pad-container {
    display: flex !important;
    justify-content: center !important;
    padding: 8px 0 !important;
}

.control-pad {
    display: grid !important;
    grid-template-columns: repeat(3, 1fr) !important;
    grid-template-rows: repeat(3, 1fr) !important;
    gap: 8px !important;
    width: 180px !important;
}

.control-btn {
    width: 52px !important;
    height: 52px !important;
    border-radius: 50% !important;
    border: 2px solid #E2E8F0 !important;
    background: white !important;
    color: #4F46E5 !important;
    font-size: 1.25rem !important;
    font-weight: bold !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

.control-btn:hover {
    background: #EEF2FF !important;
    border-color: #4F46E5 !important;
    transform: scale(1.05) !important;
}

.control-btn:active {
    transform: scale(0.95) !important;
    background: #E0E7FF !important;
}

.control-btn-stop {
    background: #FEE2E2 !important;
    color: #EF4444 !important;
    border-color: #FECACA !important;
}

.control-btn-stop:hover {
    background: #FEE2E2 !important;
    border-color: #EF4444 !important;
}

/* === EMERGENCY STOP BUTTON === */
.emergency-stop-btn {
    background: linear-gradient(135deg, #EF4444, #DC2626) !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 14px 24px !important;
    border-radius: 10px !important;
    border: 3px solid #B91C1C !important;
    box-shadow: 0 4px 14px 0 rgba(239, 68, 68, 0.4) !important;
    animation: emergency-pulse 2s infinite !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.emergency-stop-btn:hover {
    background: linear-gradient(135deg, #DC2626, #B91C1C) !important;
    box-shadow: 0 6px 20px 0 rgba(239, 68, 68, 0.5) !important;
    transform: translateY(-2px) !important;
}

.emergency-stop-btn:active {
    transform: translateY(0) !important;
}

/* === CAMERA FEED === */
.camera-container {
    border-radius: 8px !important;
    overflow: hidden !important;
    background: #1E293B !important;
}

.camera-container img {
    width: 100% !important;
    height: auto !important;
    display: block !important;
}

.camera-placeholder {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: 160px !important;
    color: #64748B !important;
    font-size: 0.875rem !important;
}

/* === CHAT STYLING === */
.chat-card {
    height: 100% !important;
}

.chat-card .chatbot {
    border: none !important;
    background: transparent !important;
}

/* === INPUT STYLING === */
.ranger-input textarea {
    border: 2px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}

.ranger-input textarea:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
    outline: none !important;
}

/* === QUICK ACTION BUTTONS === */
.quick-actions {
    display: flex !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
}

.quick-action-btn {
    font-size: 0.75rem !important;
    padding: 6px 12px !important;
    border-radius: 6px !important;
    background: #F1F5F9 !important;
    color: #475569 !important;
    border: 1px solid #E2E8F0 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

.quick-action-btn:hover {
    background: #E2E8F0 !important;
    border-color: #CBD5E1 !important;
}

/* === ANIMATIONS === */
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(1.1); }
}

@keyframes emergency-pulse {
    0%, 100% {
        box-shadow: 0 4px 14px 0 rgba(239, 68, 68, 0.4);
    }
    50% {
        box-shadow: 0 4px 20px 0 rgba(239, 68, 68, 0.6), 0 0 0 4px rgba(239, 68, 68, 0.2);
    }
}

/* === RESPONSIVE DESIGN === */
@media (max-width: 768px) {
    .ranger-header-content {
        flex-direction: column !important;
        text-align: center !important;
    }

    .ranger-header-left {
        flex-direction: column !important;
    }

    .control-pad {
        width: 150px !important;
    }

    .control-btn {
        width: 44px !important;
        height: 44px !important;
        font-size: 1rem !important;
    }
}

/* === MANUAL OUTPUT === */
.manual-output textarea {
    font-size: 0.8rem !important;
    color: #64748B !important;
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 6px !important;
}
"""


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

    def _get_header_html(self) -> str:
        """Generate header HTML with robot logo and status badge."""
        # Get connection status
        status_html = self._get_connection_status_html()

        # Use relative path for the robot avatar
        avatar_path = "file=" + str(ROBOT_AVATAR_PATH) if ROBOT_AVATAR_PATH.exists() else ""
        logo_html = f'<img src="{avatar_path}" class="ranger-logo" alt="Ranger">' if avatar_path else ""

        return f'''
        <div class="ranger-header">
            <div class="ranger-header-content">
                <div class="ranger-header-left">
                    {logo_html}
                    <div class="ranger-title">
                        <h1>Ranger Garden Robot</h1>
                        <p>Natural Language Control Interface</p>
                    </div>
                </div>
                {status_html}
            </div>
        </div>
        '''

    def _get_connection_status_html(self) -> str:
        """Generate connection status badge HTML."""
        if not ROS_AVAILABLE:
            return '<span class="status-badge status-simulation">Simulation Mode</span>'

        # Check if we have a ROS node
        if self._node is None:
            return '<span class="status-badge status-offline">Disconnected</span>'

        return '<span class="status-badge status-online">Connected</span>'

    def _get_battery_html(self) -> str:
        """Generate battery status HTML with visual progress bar."""
        interface = get_status_interface()
        level, status = interface.get_battery_level()

        if level < 0:
            return '''
            <div class="battery-container">
                <div class="battery-bar-wrapper">
                    <div class="battery-bar-fill medium" style="width: 50%;">
                        <span class="battery-text">--</span>
                    </div>
                </div>
                <div class="battery-status">Status unknown</div>
            </div>
            '''

        # Determine color class based on level
        if level >= 60:
            color_class = "high"
        elif level >= 20:
            color_class = "medium"
        else:
            color_class = "low"

        # Ensure minimum width for visibility
        display_width = max(level, 15)

        return f'''
        <div class="battery-container">
            <div class="battery-bar-wrapper">
                <div class="battery-bar-fill {color_class}" style="width: {display_width}%;">
                    <span class="battery-text">{level:.0f}%</span>
                </div>
            </div>
            <div class="battery-status">{status.title()}</div>
        </div>
        '''

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

        try:
            # For synchronous response (non-streaming)
            result = self.agent.invoke(message)
            output = result.get("output", "I couldn't process that request.")

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
        """Create the professional Gradio UI interface."""

        # Get avatar path for chatbot
        avatar_path = str(ROBOT_AVATAR_PATH) if ROBOT_AVATAR_PATH.exists() else None

        with gr.Blocks(
            title="Ranger Garden Robot",
        ) as demo:

            # === HEADER BAR ===
            gr.HTML(self._get_header_html())

            # === MAIN CONTENT ===
            with gr.Row():

                # === LEFT COLUMN: Chat Interface ===
                with gr.Column(scale=3):
                    with gr.Group(elem_classes=["ranger-card", "chat-card"]):
                        gr.HTML('<div class="ranger-card-title">💬 Chat Interface</div>')

                        chatbot = gr.Chatbot(
                            show_label=False,
                            height=460,
                            avatar_images=(None, avatar_path),
                        )

                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Type a command (e.g., 'move forward 1 meter', 'check battery')...",
                                show_label=False,
                                container=False,
                                scale=5,
                                elem_classes=["ranger-input"],
                            )
                            submit_btn = gr.Button(
                                "Send",
                                variant="primary",
                                scale=1,
                                min_width=100,
                            )

                        with gr.Row(elem_classes=["quick-actions"]):
                            clear_btn = gr.Button("Clear Chat", size="sm", variant="secondary")
                            example_btn1 = gr.Button("Check Status", size="sm", variant="secondary")
                            example_btn2 = gr.Button("Get Position", size="sm", variant="secondary")

                # === RIGHT COLUMN: Status & Controls ===
                with gr.Column(scale=1, min_width=300):

                    # --- Emergency Stop (Prominent) ---
                    stop_btn = gr.Button(
                        "🛑 EMERGENCY STOP",
                        variant="stop",
                        elem_classes=["emergency-stop-btn"],
                        size="lg",
                    )

                    # --- Status Card ---
                    with gr.Group(elem_classes=["ranger-card"]):
                        gr.HTML('<div class="ranger-card-title">📊 System Status</div>')

                        battery_html = gr.HTML(
                            value=self._get_battery_html(),
                        )

                        refresh_btn = gr.Button(
                            "Refresh Status",
                            size="sm",
                            variant="secondary",
                        )

                    # --- Camera Card ---
                    with gr.Group(elem_classes=["ranger-card"]):
                        gr.HTML('<div class="ranger-card-title">📷 Camera Feed</div>')

                        with gr.Column(elem_classes=["camera-container"]):
                            camera_image = gr.Image(
                                value=self.get_camera_image(),
                                show_label=False,
                                height=180,
                                container=False,
                            )

                        camera_refresh_btn = gr.Button(
                            "Refresh Camera",
                            size="sm",
                            variant="secondary",
                        )

                    # --- Manual Controls Card ---
                    with gr.Group(elem_classes=["ranger-card"]):
                        gr.HTML('<div class="ranger-card-title">🎮 Manual Controls</div>')

                        # Control pad using Gradio buttons in grid layout
                        with gr.Column(elem_classes=["control-pad-container"]):
                            with gr.Row():
                                gr.HTML('<div style="width: 52px;"></div>')
                                fwd_btn = gr.Button("↑", elem_classes=["control-btn"], min_width=52)
                                gr.HTML('<div style="width: 52px;"></div>')

                            with gr.Row():
                                left_btn = gr.Button("←", elem_classes=["control-btn"], min_width=52)
                                stop_manual_btn = gr.Button("■", elem_classes=["control-btn", "control-btn-stop"], min_width=52)
                                right_btn = gr.Button("→", elem_classes=["control-btn"], min_width=52)

                            with gr.Row():
                                gr.HTML('<div style="width: 52px;"></div>')
                                back_btn = gr.Button("↓", elem_classes=["control-btn"], min_width=52)
                                gr.HTML('<div style="width: 52px;"></div>')

                        manual_output = gr.Textbox(
                            show_label=False,
                            interactive=False,
                            lines=1,
                            placeholder="Control feedback...",
                            elem_classes=["manual-output"],
                        )

            # === EVENT HANDLERS ===

            # Chat submission
            submit_btn.click(
                fn=self.chat_response,
                inputs=[msg, chatbot],
                outputs=[chatbot],
            ).then(
                fn=lambda: "",
                outputs=[msg],
            )

            msg.submit(
                fn=self.chat_response,
                inputs=[msg, chatbot],
                outputs=[chatbot],
            ).then(
                fn=lambda: "",
                outputs=[msg],
            )

            # Quick action buttons
            clear_btn.click(
                fn=lambda: [],
                outputs=[chatbot],
            )

            example_btn1.click(
                fn=lambda: "What's my current status?",
                outputs=[msg],
            )

            example_btn2.click(
                fn=lambda: "What's my current position?",
                outputs=[msg],
            )

            # Emergency stop
            stop_btn.click(
                fn=self.emergency_stop,
                outputs=[manual_output],
            )

            # Status refresh (returns HTML now)
            refresh_btn.click(
                fn=self._get_battery_html,
                outputs=[battery_html],
            )

            # Camera refresh
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

        def find_available_port(start_port: int, max_tries: int) -> Optional[int]:
            for port in range(start_port, start_port + max_tries + 1):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        sock.bind(("0.0.0.0", port))
                    except OSError:
                        continue
                    return port
            return None

        port_retry_count = int(os.getenv("GRADIO_PORT_RETRY_COUNT", "5"))
        selected_port = find_available_port(self.server_port, port_retry_count)

        if selected_port is None:
            raise OSError(
                f"Cannot find empty port in range: {self.server_port}-{self.server_port + port_retry_count}"
            )

        if selected_port != self.server_port:
            logger.warning(
                "Port %s unavailable, using %s instead.",
                self.server_port,
                selected_port,
            )

        try:
            demo.launch(
                server_name="0.0.0.0",
                server_port=selected_port,
                share=self.share,
                show_error=True,
                theme=create_ranger_theme(),
                css=RANGER_CSS,
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
        default=int(os.getenv("GRADIO_SERVER_PORT") or os.getenv("GRADIO_PORT", "7860")),
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
