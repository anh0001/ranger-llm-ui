"""
Tool Registry - Consolidates all tools for the LangChain agent.

This module provides a central registry of all available tools for the
Ranger robot agent. Tools are categorized by function:
- Movement tools: MoveForward, MoveBackward, TurnAngle, StopRobot
- Status tools: BatteryStatus, SystemHealth, GetOdometry, ListNodes, ListTopics
- Perception tools: GetCameraImage

The agent's prompt will be configured to only use these tools for execution.
If the user asks for something outside this scope, the agent should refuse
or ask for clarification.
"""

import logging
from typing import Optional, Any

from langchain.tools import BaseTool

from ranger_llm_ui.tools.movement_tools import (
    MoveForwardTool,
    MoveBackwardTool,
    TurnAngleTool,
    StopRobotTool,
    get_movement_tools,
    initialize_ros_interface,
)
from ranger_llm_ui.tools.status_tools import (
    BatteryStatusTool,
    SystemHealthTool,
    GetOdometryTool,
    ListNodesTool,
    ListTopicsTool,
    get_status_tools,
    initialize_status_interface,
)
from ranger_llm_ui.tools.camera_tools import (
    GetCameraImageTool,
    get_camera_tools,
    initialize_camera_interface,
)

logger = logging.getLogger(__name__)


# Tool categories for organization and documentation
TOOL_CATEGORIES = {
    "movement": {
        "description": "Tools for controlling robot movement",
        "tools": ["MoveForward", "MoveBackward", "TurnAngle", "StopRobot"],
    },
    "status": {
        "description": "Tools for querying robot and system status",
        "tools": ["BatteryStatus", "SystemHealth", "GetOdometry"],
    },
    "diagnostics": {
        "description": "Tools for ROS 2 system diagnostics",
        "tools": ["ListNodes", "ListTopics"],
    },
    "perception": {
        "description": "Tools for camera and perception data",
        "tools": ["GetCameraImage"],
    },
}


def get_all_tools(
    include_movement: bool = True,
    include_status: bool = True,
    include_diagnostics: bool = True,
    include_perception: bool = True,
) -> list[BaseTool]:
    """
    Get all registered tools for the agent.

    Args:
        include_movement: Include movement tools (MoveForward, etc.)
        include_status: Include status tools (BatteryStatus, etc.)
        include_diagnostics: Include diagnostic tools (ListNodes, etc.)

    Returns:
        List of BaseTool instances
    """
    tools: list[BaseTool] = []

    if include_movement:
        tools.extend([
            MoveForwardTool(),
            MoveBackwardTool(),
            TurnAngleTool(),
            StopRobotTool(),
        ])

    if include_status:
        tools.extend([
            BatteryStatusTool(),
            SystemHealthTool(),
            GetOdometryTool(),
        ])

    if include_diagnostics:
        tools.extend([
            ListNodesTool(),
            ListTopicsTool(),
        ])

    if include_perception:
        tools.extend([
            GetCameraImageTool(),
        ])

    logger.info(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")
    return tools


def get_tools_by_category(category: str) -> list[BaseTool]:
    """
    Get tools for a specific category.

    Args:
        category: One of "movement", "status", "diagnostics"

    Returns:
        List of tools in that category
    """
    if category not in TOOL_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Valid: {list(TOOL_CATEGORIES.keys())}")

    if category == "movement":
        return get_movement_tools()
    elif category == "status":
        return [BatteryStatusTool(), SystemHealthTool(), GetOdometryTool()]
    elif category == "diagnostics":
        return [ListNodesTool(), ListTopicsTool()]
    elif category == "perception":
        return get_camera_tools()
    else:
        return []


def initialize_all_tools(node: Optional[Any] = None) -> list[BaseTool]:
    """
    Initialize all tools with a ROS 2 node and return them.

    This should be called once when setting up the agent to ensure
    all tools have access to the ROS 2 node for communication.

    Args:
        node: ROS 2 node instance (or None for simulation mode)

    Returns:
        List of initialized tools
    """
    # Initialize ROS interfaces
    initialize_ros_interface(node)
    initialize_status_interface(node)
    initialize_camera_interface(node)

    # Return all tools
    return get_all_tools()


def get_tool_descriptions() -> str:
    """
    Get formatted descriptions of all available tools.

    Returns:
        Markdown-formatted string describing all tools
    """
    lines = ["# Available Robot Tools\n"]

    for category, info in TOOL_CATEGORIES.items():
        lines.append(f"## {category.title()}")
        lines.append(f"{info['description']}\n")

        tools = get_tools_by_category(category)
        for tool in tools:
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}\n")

    return "\n".join(lines)


def get_tool_names() -> list[str]:
    """Get list of all tool names."""
    tools = get_all_tools()
    return [tool.name for tool in tools]


def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """
    Get a specific tool by name.

    Args:
        name: Tool name (case-sensitive)

    Returns:
        Tool instance or None if not found
    """
    tools = get_all_tools()
    for tool in tools:
        if tool.name == name:
            return tool
    return None


# Safety-critical tools that should always be available
SAFETY_CRITICAL_TOOLS = ["StopRobot"]


def get_safety_tools() -> list[BaseTool]:
    """Get only the safety-critical tools (like emergency stop)."""
    return [StopRobotTool()]


# Tool usage statistics (for monitoring)
class ToolUsageTracker:
    """Track tool usage for monitoring and analytics."""

    def __init__(self):
        self._usage_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}

    def record_usage(self, tool_name: str, success: bool = True):
        """Record a tool usage."""
        self._usage_counts[tool_name] = self._usage_counts.get(tool_name, 0) + 1
        if not success:
            self._error_counts[tool_name] = self._error_counts.get(tool_name, 0) + 1

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_calls": sum(self._usage_counts.values()),
            "by_tool": dict(self._usage_counts),
            "errors": dict(self._error_counts),
        }


# Global usage tracker
_usage_tracker = ToolUsageTracker()


def get_usage_tracker() -> ToolUsageTracker:
    """Get the global tool usage tracker."""
    return _usage_tracker
