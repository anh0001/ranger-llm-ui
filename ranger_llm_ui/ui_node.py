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
import re
from datetime import datetime
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
    """Create Ranger Agent style theme with clean, centered design."""
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="gray",
        spacing_size="md",
        radius_size="lg",
        text_size="md",
    ).set(
        # Primary button styling
        button_primary_background_fill="#4A90E2",
        button_primary_background_fill_hover="#2F6FB2",
        button_primary_text_color="white",
        button_primary_border_color="#4A90E2",
        button_primary_shadow="0 2px 4px rgba(74, 144, 226, 0.25)",
        # Secondary button styling
        button_secondary_background_fill="#F4F7FB",
        button_secondary_background_fill_hover="#E5ECF5",
        button_secondary_text_color="#3A4A5E",
        # Stop button styling
        button_cancel_background_fill="#EF4444",
        button_cancel_background_fill_hover="#DC2626",
        button_cancel_text_color="white",
        button_cancel_border_color="#EF4444",
        # Block/card styling - cleaner for single column
        block_background_fill="white",
        block_border_color="transparent",
        block_shadow="none",
        block_title_text_weight="600",
        block_label_text_weight="500",
        # Input styling
        input_background_fill="white",
        input_border_color="#E3E8EF",
        input_border_color_focus="#4A90E2",
        # Body/container - light background
        body_background_fill="#F6F8FB",
    )


# CSS for Ranger Agent style interface
RANGER_CSS = """
/* ================================================
   RANGER ROBOT - SPOT AGENT STYLE INTERFACE
   ================================================ */

@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Sora:wght@400;600;700&display=swap');

:root {
    --rg-primary: #4A90E2;
    --rg-primary-strong: #2F6FB2;
    --rg-ink: #1E2A3B;
    --rg-muted: #5B6B7A;
    --rg-border: #E3E8EF;
    --rg-panel: #FFFFFF;
    --rg-panel-soft: #F4F7FB;
    --rg-query: #EAF4FF;
    --rg-response: #FFF6E5;
}

/* Gradio v4 renders inside a shadow root; :host targets the app wrapper. */
:host,
:host([data-color-mode="dark"]) {
    color-scheme: light !important;
    background: #F6F8FB !important;
    display: block !important;
    min-height: 100vh !important;
}

.gradio-container {
    font-family: "Sora", "Segoe UI", sans-serif !important;
    color: var(--rg-ink) !important;
    background: #F6F8FB !important;
    min-height: 100vh !important;
}

/* === HIDE GRADIO FOOTER === */
footer { display: none !important; }

/* === HIDE GRADIO MENU BUTTON (...) === */
.gradio-container button[aria-label="Show actions"],
.gradio-container button[aria-label="Open actions menu"],
.gradio-container .actions-menu-toggle,
.gradio-container [data-testid="actions-menu"] { 
    display: none !important; 
}

/* === HIDE IMAGE ACTION BUTTONS (fullscreen, download, share) === */
.spot-mascot-image .image-button-row,
.spot-mascot-image [class*="image-button"],
.spot-mascot-image button[aria-label*="fullscreen"],
.spot-mascot-image button[aria-label*="download"],
.spot-mascot-image button[aria-label*="share"],
.spot-mascot .image-container button,
.spot-mascot [class*="icon-button"],
.spot-mascot .svelte-1pijsyv button {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* === MAIN CONTAINER - Centered single column === */
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 24px 40px 32px 40px !important;
    box-sizing: border-box !important;
    background: #F6F8FB !important;
}

/* === HEADER - Centered with nav links === */
.spot-header {
    text-align: center !important;
    padding: 16px 0 10px 0 !important;
    border-top: 3px solid var(--rg-primary) !important;
    margin-bottom: 10px !important;
}

.spot-title {
    font-size: 2rem !important;
    font-weight: 700 !important;
    font-family: "Fraunces", "Sora", serif !important;
    color: var(--rg-ink) !important;
    margin: 0 0 12px 0 !important;
}

.spot-subtitle {
    font-size: 0.9rem !important;
    color: var(--rg-muted) !important;
    margin: 0 0 15px 0 !important;
}

.spot-nav {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 8px !important;
    margin-top: 10px !important;
}

.spot-nav-link {
    color: var(--rg-muted) !important;
    text-decoration: none !important;
    font-size: 1rem !important;
    cursor: pointer !important;
    padding: 8px 16px !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    background: transparent !important;
    border: none !important;
}

.spot-nav-link:hover {
    color: var(--rg-primary-strong) !important;
    background: #EAF2FB !important;
}

.spot-nav-link.active {
    color: var(--rg-primary-strong) !important;
    font-weight: 600 !important;
    background: #EAF2FB !important;
}

.spot-nav-separator {
    color: #C7D2E2 !important;
    font-size: 1rem !important;
}

/* === ROBOT MASCOT - Larger centered image === */
.spot-mascot {
    display: flex !important;
    justify-content: center !important;
    padding: 20px 0 !important;
}

.spot-mascot img,
.spot-mascot-image img {
    width: 190px !important;
    height: auto !important;
    object-fit: contain !important;
    border-radius: 12px !important;
}

.spot-mascot-image {
    max-width: 220px !important;
    margin: 0 auto !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* === CHAT INPUT AREA - Cyan/teal background === */
.spot-input-container {
    background: var(--rg-panel) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    margin: 15px 0 !important;
    border: 1px solid var(--rg-border) !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
}

.spot-input-row {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

.spot-attach-btn {
    min-width: 44px !important;
    width: 44px !important;
    height: 44px !important;
    border-radius: 50% !important;
    background: var(--rg-panel) !important;
    border: 2px solid #D3DCE8 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.2rem !important;
    transition: all 0.2s ease !important;
    padding: 0 !important;
}

.spot-attach-btn:hover {
    border-color: var(--rg-primary) !important;
    background: #EAF2FB !important;
}

.spot-input-field textarea {
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    background: var(--rg-panel) !important;
    font-size: 1rem !important;
    color: var(--rg-ink) !important;
    caret-color: var(--rg-primary-strong) !important;
    min-height: 46px !important;
    line-height: 1.4 !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
}

.spot-input-field textarea::placeholder {
    color: #9AA7B5 !important;
}

.spot-input-field textarea:focus {
    outline: none !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.1), 0 0 0 2px rgba(74, 144, 226, 0.2) !important;
}

.spot-send-btn {
    padding: 12px 24px !important;
    border-radius: 8px !important;
    background: #E5ECF5 !important;
    color: #4B5B6C !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    font-weight: 600 !important;
    min-width: 80px !important;
}

.spot-send-btn:hover {
    background: var(--rg-primary) !important;
    color: white !important;
}

/* === MESSAGE CARDS === */
.spot-messages-container {
    margin-top: 15px !important;
}

.spot-message-card {
    background: var(--rg-panel) !important;
    border: 1px solid var(--rg-border) !important;
    border-radius: 12px !important;
    margin: 15px 0 !important;
    overflow: hidden !important;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.05) !important;
}

.spot-message-header {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    padding: 10px 16px !important;
    background: #F9FBFD !important;
    border-bottom: 1px solid #E6ECF4 !important;
}

.spot-message-header-text {
    font-weight: 600 !important;
    color: var(--rg-ink) !important;
}

.spot-message-actions {
    display: flex !important;
    gap: 8px !important;
}

.spot-action-btn {
    width: 28px !important;
    height: 28px !important;
    border: none !important;
    background: transparent !important;
    cursor: pointer !important;
    border-radius: 4px !important;
    font-size: 1rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.2s ease !important;
}

.spot-action-btn:hover {
    background: #EEF3FA !important;
}

.spot-action-btn.play-btn {
    color: var(--rg-primary-strong) !important;
}

.spot-action-btn.delete-btn {
    color: #EF4444 !important;
}

.spot-query-section {
    padding: 12px 16px !important;
    background: var(--rg-query) !important;
    border-bottom: 1px solid #D7E9FF !important;
}

.spot-query-label {
    font-weight: 600 !important;
    color: var(--rg-ink) !important;
    margin-bottom: 6px !important;
    font-size: 0.9rem !important;
}

.spot-query-text {
    color: #2E3A49 !important;
    font-size: 0.95rem !important;
}

.spot-response-section {
    padding: 12px 16px !important;
    background: var(--rg-response) !important;
}

.spot-response-label {
    font-weight: 600 !important;
    color: var(--rg-ink) !important;
    margin-bottom: 6px !important;
    font-size: 0.9rem !important;
}

.spot-response-content {
    color: #2E3A49 !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

.spot-response-image-label {
    color: var(--rg-muted) !important;
    font-size: 0.85rem !important;
    margin-bottom: 8px !important;
}

.spot-message-image {
    max-width: 100% !important;
    border-radius: 8px !important;
    margin: 10px 0 !important;
    border: 1px solid var(--rg-border) !important;
}

.spot-timestamp {
    font-size: 0.75rem !important;
    color: #8A96A6 !important;
    padding: 8px 16px !important;
    background: #F9FBFD !important;
    border-top: 1px solid #E6ECF4 !important;
}

/* === HIDE DEFAULT GRADIO TAB BUTTONS === */
.tab-nav,
#main-tabs .tab-nav,
#main-tabs > .tab-nav,
#main-tabs > div > .tab-nav,
#main-tabs [role="tablist"],
#main-tabs button[role="tab"] {
    display: none !important;
}

/* === STATUS TAB STYLING === */
.status-card {
    background: var(--rg-panel) !important;
    border: 1px solid var(--rg-border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
}

.status-card-title {
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    color: #4A5A6A !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid var(--rg-border) !important;
}

/* === BATTERY INDICATOR === */
.battery-container {
    padding: 8px 0 !important;
}

.battery-bar-wrapper {
    width: 100% !important;
    height: 24px !important;
    background: #E5ECF5 !important;
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
    color: #6B7B8C !important;
    text-align: center !important;
    margin-top: 6px !important;
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

/* === CAMERA FEED === */
.camera-container {
    border-radius: 8px !important;
    overflow: hidden !important;
    background: #E5ECF5 !important;
}

.camera-container img {
    width: 100% !important;
    height: auto !important;
    display: block !important;
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
    background: rgba(16, 185, 129, 0.15) !important;
    color: #059669 !important;
}

.status-online::before {
    background: #10B981;
    animation: pulse-dot 2s infinite;
}

.status-simulation {
    background: rgba(245, 158, 11, 0.15) !important;
    color: #D97706 !important;
}

.status-simulation::before {
    background: #F59E0B;
}

.status-offline {
    background: rgba(239, 68, 68, 0.15) !important;
    color: #DC2626 !important;
}

.status-offline::before {
    background: #EF4444;
}

/* === FOOTER === */
.spot-footer {
    text-align: center !important;
    padding: 30px 0 !important;
    color: #6B7B8C !important;
    font-size: 0.875rem !important;
    border-top: 1px solid var(--rg-border) !important;
    margin-top: 20px !important;
}

/* === QUICK ACTION BUTTONS === */
.quick-actions {
    display: none !important;
}

/* === SETTINGS TAB === */
.settings-section {
    background: var(--rg-panel) !important;
    border: 1px solid var(--rg-border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-bottom: 15px !important;
}

.settings-title {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--rg-ink) !important;
    margin-bottom: 15px !important;
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
    .gradio-container {
        padding: 16px !important;
    }

    .spot-title {
        font-size: 1.5rem !important;
    }

    .spot-mascot img {
        width: 150px !important;
    }

    .spot-nav {
        flex-wrap: wrap !important;
    }

    .spot-input-row {
        flex-wrap: wrap !important;
    }
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
                        <h1>Robot Garden Assistant</h1>
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

    def _get_spot_header_html(self) -> str:
        """Generate Spot Agent style header with navigation links."""
        status_html = self._get_connection_status_html()

        return f'''
        <div class="spot-header">
            <h1 class="spot-title">Robot Garden Assistant</h1>
            <nav class="spot-nav" id="spot-nav">
                <button class="spot-nav-link active" id="nav-home">Home</button>
                <span class="spot-nav-separator">|</span>
                <button class="spot-nav-link" id="nav-status">Status</button>
                <span class="spot-nav-separator">|</span>
                <button class="spot-nav-link" id="nav-settings">Settings</button>
            </nav>
        </div>
        <script>
            function getGradioRoot() {{
                const app = document.querySelector('gradio-app');
                if (app && app.shadowRoot) {{
                    return app.shadowRoot;
                }}
                return document;
            }}

            function setActiveNav(index) {{
                const root = getGradioRoot();
                root.querySelectorAll('.spot-nav-link').forEach((link, i) => {{
                    if (i === index) {{
                        link.classList.add('active');
                    }} else {{
                        link.classList.remove('active');
                    }}
                }});
            }}

            function switchToTab(index) {{
                const root = getGradioRoot();
                const tabButtons = root.querySelectorAll('#main-tabs .tab-nav button, #main-tabs [role="tablist"] button, #main-tabs button[role="tab"]');
                if (tabButtons[index]) {{
                    tabButtons[index].click();
                }}
                setActiveNav(index);
            }}

            function syncNavToTabs() {{
                const root = getGradioRoot();
                const tabButtons = root.querySelectorAll('#main-tabs .tab-nav button, #main-tabs [role="tablist"] button, #main-tabs button[role="tab"]');
                tabButtons.forEach((btn, i) => {{
                    if (btn.getAttribute('aria-selected') === 'true' || btn.classList.contains('selected')) {{
                        setActiveNav(i);
                    }}
                }});
            }}

            function attachNavHandlers() {{
                const root = getGradioRoot();
                const navButtons = root.querySelectorAll('.spot-nav-link');
                navButtons.forEach((btn, i) => {{
                    btn.addEventListener('click', () => switchToTab(i));
                }});
            }}

            setTimeout(() => {{
                const app = document.querySelector('gradio-app');
                if (app) {{
                    app.setAttribute('data-color-mode', 'light');
                }}
                attachNavHandlers();
                syncNavToTabs();
            }}, 0);
        </script>
        '''

    def _get_mascot_html(self) -> str:
        """Generate larger centered robot mascot image."""
        avatar_path = "file=" + str(ROBOT_AVATAR_PATH) if ROBOT_AVATAR_PATH.exists() else ""
        if avatar_path:
            return f'''
            <div class="spot-mascot">
                <img src="{avatar_path}" alt="Ranger Robot">
            </div>
            '''
        return '<div class="spot-mascot"><p>Robot Avatar</p></div>'

    def _get_footer_html(self) -> str:
        """Generate ROSA branding footer."""
        current_year = datetime.now().year
        return f'''
        <div class="spot-footer">
            Ranger Garden Assistant
        </div>
        '''

    def _render_messages_html(self, history: list[dict]) -> str:
        """
        Render message history as custom HTML cards.

        Args:
            history: List of message dicts with 'user', 'assistant', 'timestamp', and optional 'image'

        Returns:
            HTML string of message cards
        """
        if not history:
            return '<div class="spot-messages-container"><p style="text-align: center; color: #999; padding: 20px;">No messages yet. Start a conversation!</p></div>'

        html_parts = ['<div class="spot-messages-container">']

        for idx, msg in enumerate(history):
            user_content = msg.get("user", "")
            assistant_content = msg.get("assistant", "")
            timestamp = msg.get("timestamp", "")
            image_data = msg.get("image", None)

            # Build image HTML if present
            image_html = ""
            if image_data:
                image_html = f'''
                <div class="spot-response-image-label">RGB + Depth</div>
                <img src="{image_data}" class="spot-message-image" alt="Camera image">
                '''

            card_html = f'''
            <div class="spot-message-card" data-index="{idx}">
                <div class="spot-message-header">
                    <span class="spot-message-header-text">Response</span>
                    <div class="spot-message-actions">
                        <button class="spot-action-btn play-btn" title="Play (TTS)">▶</button>
                        <button class="spot-action-btn delete-btn" title="Delete">🗑</button>
                    </div>
                </div>
                <div class="spot-query-section">
                    <div class="spot-query-label">Query:</div>
                    <div class="spot-query-text">{user_content}</div>
                </div>
                <div class="spot-response-section">
                    <div class="spot-response-label">Response:</div>
                    {image_html}
                    <div class="spot-response-content">{assistant_content}</div>
                </div>
                <div class="spot-timestamp">Timestamp: {timestamp}</div>
            </div>
            '''
            html_parts.append(card_html)

        html_parts.append('</div>')
        return "".join(html_parts)

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
        """Create the Spot Agent style Gradio UI interface."""

        with gr.Blocks(
            title="Robot Garden Assistant",
        ) as demo:

            # === HEADER WITH NAV LINKS ===
            gr.HTML(self._get_spot_header_html())

            # === ROBOT MASCOT ===
            with gr.Row(elem_classes=["spot-mascot"]):
                mascot_value = str(ROBOT_AVATAR_PATH) if ROBOT_AVATAR_PATH.exists() else None
                gr.Image(
                    value=mascot_value,
                    show_label=False,
                    interactive=False,
                    container=False,
                    elem_classes=["spot-mascot-image"],
                )

            # === TABBED CONTENT (hidden native tab buttons) ===
            with gr.Tabs(elem_id="main-tabs") as tabs:

                # === HOME TAB - Chat Interface ===
                with gr.Tab("Home", id=0):

                    # Chat input area with cyan background
                    with gr.Group(elem_classes=["spot-input-container"]):
                        with gr.Row(elem_classes=["spot-input-row"]):
                            msg = gr.Textbox(
                                placeholder="all right can you describe what you see in the camera",
                                show_label=False,
                                container=False,
                                scale=5,
                                elem_classes=["spot-input-field"],
                            )
                            submit_btn = gr.Button(
                                "Send",
                                elem_classes=["spot-send-btn"],
                                min_width=80,
                            )

                    # Quick action buttons
                    with gr.Row(elem_classes=["quick-actions"]):
                        clear_btn = gr.Button("Clear Chat", size="sm", variant="secondary")
                        example_btn1 = gr.Button("Check Status", size="sm", variant="secondary")
                        example_btn2 = gr.Button("Get Camera", size="sm", variant="secondary")

                    # Message history state (for card-based display)
                    message_history = gr.State([])

                    # Messages display (custom HTML cards)
                    messages_html = gr.HTML(
                        value=self._render_messages_html([]),
                        elem_classes=["spot-messages-container"],
                    )

                # === STATUS TAB ===
                with gr.Tab("Status", id=1):

                    # Emergency Stop (Prominent)
                    stop_btn = gr.Button(
                        "🛑 EMERGENCY STOP",
                        variant="stop",
                        elem_classes=["emergency-stop-btn"],
                        size="lg",
                    )
                    stop_output = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        lines=1,
                        placeholder="Emergency stop feedback...",
                        visible=True,
                    )

                    # Battery Status Card
                    with gr.Group(elem_classes=["status-card"]):
                        gr.HTML('<div class="status-card-title">📊 Battery Status</div>')
                        battery_html = gr.HTML(
                            value=self._get_battery_html(),
                        )
                        refresh_btn = gr.Button(
                            "Refresh Status",
                            size="sm",
                            variant="secondary",
                        )

                    # Camera Feed Card
                    with gr.Group(elem_classes=["status-card"]):
                        gr.HTML('<div class="status-card-title">📷 Camera Feed</div>')
                        with gr.Column(elem_classes=["camera-container"]):
                            camera_image = gr.Image(
                                value=self.get_camera_image(),
                                show_label=False,
                                height=240,
                                container=False,
                            )
                        camera_refresh_btn = gr.Button(
                            "Refresh Camera",
                            size="sm",
                            variant="secondary",
                        )

                # === SETTINGS TAB ===
                with gr.Tab("Settings", id=2):
                    with gr.Group(elem_classes=["settings-section"]):
                        gr.HTML('<div class="settings-title">⚙️ Configuration</div>')
                        gr.Markdown("""
                        **LLM Provider:** Configure your language model settings

                        Settings functionality coming soon. For now, configure via:
                        - Environment variables (.env file)
                        - Command line arguments

                        See documentation for details.
                        """)

                    with gr.Group(elem_classes=["settings-section"]):
                        gr.HTML('<div class="settings-title">📡 Connection</div>')
                        connection_status = gr.HTML(
                            value=self._get_connection_status_html(),
                        )
                        gr.Button("Refresh Connection", size="sm", variant="secondary")

            # === FOOTER ===
            gr.HTML(self._get_footer_html())

            # === EVENT HANDLERS ===

            # Chat submission - uses new card-based display
            def process_chat(message: str, history: list[dict]):
                """Process chat and return updated history with HTML."""
                if not message.strip():
                    return "", history, self._render_messages_html(history)

                if not self.agent:
                    timestamp = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
                    new_entry = {
                        "user": message,
                        "assistant": "Agent not initialized. Please check configuration.",
                        "timestamp": timestamp,
                        "image": None,
                    }
                    history.append(new_entry)
                    return "", history, self._render_messages_html(history)

                try:
                    # Get response from agent
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

                    # Optional: Show token usage
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

                    # Extract image data if present (from camera tool)
                    image_data = None
                    if "data:image" in output:
                        match = re.search(r'!\[.*?\]\((data:image[^)]+)\)', output)
                        if match:
                            image_data = match.group(1)
                            # Remove image markdown from text output
                            output = re.sub(r'!\[.*?\]\(data:image[^)]+\)', '', output).strip()

                    # Create timestamp
                    timestamp = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")

                    # Add to history
                    new_entry = {
                        "user": message,
                        "assistant": output,
                        "timestamp": timestamp,
                        "image": image_data,
                    }
                    history.append(new_entry)

                    return "", history, self._render_messages_html(history)

                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    timestamp = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
                    new_entry = {
                        "user": message,
                        "assistant": f"Error: {str(e)}",
                        "timestamp": timestamp,
                        "image": None,
                    }
                    history.append(new_entry)
                    return "", history, self._render_messages_html(history)

            submit_btn.click(
                fn=process_chat,
                inputs=[msg, message_history],
                outputs=[msg, message_history, messages_html],
            )

            msg.submit(
                fn=process_chat,
                inputs=[msg, message_history],
                outputs=[msg, message_history, messages_html],
            )

            # Clear chat
            def clear_chat():
                return [], self._render_messages_html([])

            clear_btn.click(
                fn=clear_chat,
                outputs=[message_history, messages_html],
            )

            # Quick action buttons
            example_btn1.click(
                fn=lambda: "What's my current status?",
                outputs=[msg],
            )

            example_btn2.click(
                fn=lambda: "Show me what the camera sees",
                outputs=[msg],
            )

            # Emergency stop
            stop_btn.click(
                fn=self.emergency_stop,
                outputs=[stop_output],
            )

            # Status refresh
            refresh_btn.click(
                fn=self._get_battery_html,
                outputs=[battery_html],
            )

            # Camera refresh
            camera_refresh_btn.click(
                fn=self.get_camera_image,
                outputs=[camera_image],
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
