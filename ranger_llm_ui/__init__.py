"""
Ranger LLM UI - LLM-driven natural language operator interface for Ranger robot.

This package provides a Gradio-based chat UI integrated with a LangChain agent
to interpret natural language commands and execute safe ROS 2 actions on the
Ranger robot.

Components:
- ui_node: Main entry point that starts rclpy and Gradio app
- agent_interface: LangChain agent integration
- tools: Ranger-specific tool implementations (movement, status, etc.)
- schemas: Command dataclasses and JSON schemas
- safety: Safety checks and velocity limits
- utils: Logging and configuration utilities
"""

__version__ = "0.1.0"
__author__ = "Anh Nguyen"

from ranger_llm_ui.agent_interface import RangerAgent
from ranger_llm_ui.tools.all_tools import get_all_tools

__all__ = [
    "RangerAgent",
    "get_all_tools",
    "__version__",
]
