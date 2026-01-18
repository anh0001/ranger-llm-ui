"""
Ranger Tools - Registry of vetted tool functions for robot operations.

Each tool is a deterministic function call that wraps a specific ROS 2 action
or query. Tools are restricted to pre-defined ROS actions/services/topics,
preventing arbitrary commands outside the allowed set.

Tool Categories:
- Movement Tools: move_forward, move_backward, turn_angle, stop
- Status Tools: battery_status, system_health, sensor_status
- Diagnostics Tools: list_nodes, list_topics
- Perception Tools: get_camera_image
"""

from ranger_llm_ui.tools.movement_tools import (
    MoveForwardTool,
    MoveBackwardTool,
    TurnAngleTool,
    StopRobotTool,
)
from ranger_llm_ui.tools.status_tools import (
    BatteryStatusTool,
    SystemHealthTool,
    GetOdometryTool,
)
from ranger_llm_ui.tools.camera_tools import GetCameraImageTool
from ranger_llm_ui.tools.all_tools import get_all_tools

__all__ = [
    "MoveForwardTool",
    "MoveBackwardTool",
    "TurnAngleTool",
    "StopRobotTool",
    "BatteryStatusTool",
    "SystemHealthTool",
    "GetOdometryTool",
    "GetCameraImageTool",
    "get_all_tools",
]
