"""
Status Tools - Robot status and telemetry tools for the LangChain agent.

These tools query robot state and ROS 2 system information:
- BatteryStatus: Get current battery level
- SystemHealth: Check if all essential nodes are running
- GetOdometry: Get current robot position
- ListNodes: List active ROS 2 nodes
- ListTopics: List active ROS 2 topics
"""

import time
import logging
import math
from typing import Optional, Type, Any

from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from ranger_llm_ui.utils.logger import log_tool_call
from ranger_llm_ui.safety.guard import get_safety_guard

logger = logging.getLogger(__name__)

# Try to import ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import BatteryState
    from nav_msgs.msg import Odometry
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning("ROS 2 (rclpy) not available. Running in simulation mode.")


class ROSStatusInterface:
    """
    Singleton interface for ROS 2 status queries.

    Manages subscriptions to status topics and provides query methods.
    """

    _instance: Optional["ROSStatusInterface"] = None
    _node: Optional[Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, node: Optional[Any] = None):
        """Initialize the status interface with a node."""
        # Allow re-initialization if a real node is now available but we previously ran in simulation
        if self._initialized and not (self._simulation_mode and node is not None):
            return

        self._node = node
        self._battery_state: Optional[Any] = None
        self._odom: Optional[Any] = None
        self._simulation_mode = not ROS_AVAILABLE or node is None

        # Simulated values for testing
        self._sim_battery_level = 85.0
        self._sim_position = (0.0, 0.0, 0.0)

        if not self._simulation_mode and self._node is not None:
            # Subscribe to battery state
            self._battery_sub = self._node.create_subscription(
                BatteryState,
                "/battery_state",
                self._battery_callback,
                10
            )
            # Subscribe to odometry
            self._odom_sub = self._node.create_subscription(
                Odometry,
                "/odom",
                self._odom_callback,
                10
            )
            logger.info("ROS status interface initialized with node")
        else:
            logger.info("ROS status interface running in simulation mode")

        self._initialized = True

    def _battery_callback(self, msg):
        """Callback for battery state messages."""
        self._battery_state = msg

    def _odom_callback(self, msg):
        """Callback for odometry messages."""
        self._odom = msg

    @property
    def simulation_mode(self) -> bool:
        return self._simulation_mode

    def get_battery_level(self) -> tuple[float, str, float]:
        """
        Get current battery level.

        Returns:
            Tuple of (battery_percentage, status_string, voltage_v)
        """
        if self._simulation_mode:
            level = self._sim_battery_level
            status = "charging" if level < 100 else "full"
            return level, status, 0.0

        if self._battery_state is None:
            return -1.0, "unknown", 0.0

        # Some integrations publish decivolts (e.g. 490 => 49.0V), others publish volts.
        voltage_raw = float(getattr(self._battery_state, "voltage", 0.0) or 0.0)
        voltage_v = voltage_raw / 10.0 if voltage_raw > 100.0 else voltage_raw

        # BatteryState.percentage is commonly 0..1, but some systems publish 0..100.
        try:
            level_raw = float(getattr(self._battery_state, "percentage", float("nan")))
        except (TypeError, ValueError):
            level_raw = float("nan")
        level = -1.0
        if not math.isnan(level_raw):
            if 0.0 <= level_raw <= 1.0:
                level = level_raw * 100.0
            elif 0.0 <= level_raw <= 100.0:
                level = level_raw

        # Determine status from power_supply_status
        # sensor_msgs/BatteryState: 0=UNKNOWN, 1=CHARGING, 2=DISCHARGING, 3=NOT_CHARGING, 4=FULL
        status_map = {0: "unknown", 1: "charging", 2: "discharging", 3: "not charging", 4: "full"}
        status = status_map.get(self._battery_state.power_supply_status, "unknown")

        return level, status, voltage_v

    def get_odometry(self) -> Optional[dict]:
        """
        Get current odometry data.

        Returns:
            Dictionary with position (x, y, z) and orientation (yaw)
        """
        if self._simulation_mode:
            x, y, yaw = self._sim_position
            return {
                "position": {"x": x, "y": y, "z": 0.0},
                "orientation_yaw_deg": yaw * 57.2958,
                "linear_velocity": 0.0,
                "angular_velocity": 0.0,
            }

        if self._odom is None:
            return None

        import math

        pos = self._odom.pose.pose.position
        ori = self._odom.pose.pose.orientation
        vel = self._odom.twist.twist

        # Convert quaternion to yaw
        siny_cosp = 2 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1 - 2 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return {
            "position": {"x": pos.x, "y": pos.y, "z": pos.z},
            "orientation_yaw_deg": yaw * 57.2958,
            "linear_velocity": vel.linear.x,
            "angular_velocity": vel.angular.z,
        }

    def list_nodes(self) -> list[str]:
        """List active ROS 2 nodes."""
        if self._simulation_mode:
            return [
                "/ranger_llm_ui",
                "/robot_state_publisher",
                "/motor_controller",
                "/battery_monitor",
                "/sensor_fusion",
            ]

        if self._node is None:
            return []

        try:
            node_names = self._node.get_node_names()
            return sorted(node_names)
        except Exception as e:
            logger.error(f"Error listing nodes: {e}")
            return []

    def list_topics(self) -> list[tuple[str, list[str]]]:
        """List active ROS 2 topics with their types."""
        if self._simulation_mode:
            return [
                ("/cmd_vel", ["geometry_msgs/msg/Twist"]),
                ("/odom", ["nav_msgs/msg/Odometry"]),
                ("/battery_state", ["sensor_msgs/msg/BatteryState"]),
                ("/camera/image_raw", ["sensor_msgs/msg/Image"]),
                ("/scan", ["sensor_msgs/msg/LaserScan"]),
                ("/tf", ["tf2_msgs/msg/TFMessage"]),
            ]

        if self._node is None:
            return []

        try:
            topics = self._node.get_topic_names_and_types()
            return sorted(topics)
        except Exception as e:
            logger.error(f"Error listing topics: {e}")
            return []

    def check_essential_topics(self) -> dict[str, bool]:
        """Check if essential topics are active."""
        essential_topics = ["/cmd_vel", "/odom", "/battery_state"]

        if self._simulation_mode:
            return {topic: True for topic in essential_topics}

        active_topics = [t[0] for t in self.list_topics()]
        return {topic: topic in active_topics for topic in essential_topics}


# Global status interface instance
_status_interface: Optional[ROSStatusInterface] = None


def get_status_interface() -> ROSStatusInterface:
    """Get or create the ROS status interface singleton."""
    global _status_interface
    if _status_interface is None:
        _status_interface = ROSStatusInterface()
    # Only auto-initialize (simulation mode) if not yet initialized and no node is expected.
    # Prefer explicit initialization via initialize_status_interface(node).
    if not _status_interface._initialized:
        _status_interface.initialize(None)
    return _status_interface


def initialize_status_interface(node: Optional[Any] = None):
    """Initialize the status interface with a node."""
    interface = get_status_interface()
    interface.initialize(node)
    return interface


# Tool Input Schemas
class EmptyInput(BaseModel):
    """Empty input schema for tools that don't require parameters."""
    pass


class BatteryStatusTool(BaseTool):
    """Tool to get current battery level."""

    name: str = "BatteryStatus"
    description: str = (
        "Get the current battery level and charging status of the robot. "
        "Use this when you need to check how much battery the robot has left."
    )
    args_schema: Type[BaseModel] = EmptyInput
    return_direct: bool = False

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get battery status."""
        start_time = time.time()

        interface = get_status_interface()
        level, status, voltage_v = interface.get_battery_level()

        if level < 0 and voltage_v <= 0:
            result = "Battery status unavailable"
            success = False
        else:
            parts = []
            if voltage_v > 0:
                parts.append(f"Voltage: {voltage_v:.1f} V")
            if level >= 0:
                parts.append(f"{level:.1f}%")
            parts.append(f"({status})")
            result = "Battery: " + " ".join(parts)
            success = True

            # Update safety guard with battery level if percentage is available
            if level >= 0:
                safety_guard = get_safety_guard()
                safety_guard.update_battery_level(level)

            # Add warnings if applicable (check critical first)
            if 0 <= level < 10:
                result += " - CRITICAL: Battery very low!"
            elif 0 <= level < 20:
                result += " - WARNING: Low battery!"

        execution_time = (time.time() - start_time) * 1000
        log_tool_call(
            tool_name=self.name,
            parameters={},
            result=result,
            success=success,
            execution_time_ms=execution_time,
        )

        return result


class SystemHealthTool(BaseTool):
    """Tool to check overall system health."""

    name: str = "SystemHealth"
    description: str = (
        "Check the overall health of the robot system. "
        "This checks if all essential ROS nodes and topics are running properly. "
        "Use this to diagnose issues or verify the robot is ready to operate."
    )
    args_schema: Type[BaseModel] = EmptyInput
    return_direct: bool = False

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Check system health."""
        start_time = time.time()

        interface = get_status_interface()

        # Check essential topics
        topic_status = interface.check_essential_topics()
        all_topics_ok = all(topic_status.values())

        # Get node count
        nodes = interface.list_nodes()
        node_count = len(nodes)

        # Build result
        lines = ["System Health Report:"]

        if interface.simulation_mode:
            lines.append("  Mode: SIMULATION")

        lines.append(f"  Active nodes: {node_count}")

        lines.append("  Essential topics:")
        for topic, is_active in topic_status.items():
            status_str = "OK" if is_active else "MISSING"
            lines.append(f"    {topic}: {status_str}")

        # Get battery status
        level, _, _voltage = interface.get_battery_level()
        if level >= 0:
            lines.append(f"  Battery: {level:.1f}%")

        # Overall status
        # Battery unavailable (level < 0) means no message received yet — not a hard failure
        battery_critical = 0 <= level <= 10
        if all_topics_ok and not battery_critical:
            lines.append("  Overall: HEALTHY - Robot is ready to operate")
            success = True
        else:
            issues = []
            if not all_topics_ok:
                missing = [t for t, ok in topic_status.items() if not ok]
                issues.append(f"Missing topics: {', '.join(missing)}")
            if battery_critical:
                issues.append("Critical battery level")
            if not issues:
                issues.append("Unknown health issue")
            lines.append(f"  Overall: ISSUES DETECTED - {'; '.join(issues)}")
            success = False

        result = "\n".join(lines)

        execution_time = (time.time() - start_time) * 1000
        log_tool_call(
            tool_name=self.name,
            parameters={},
            result=result,
            success=success,
            execution_time_ms=execution_time,
        )

        return result


class GetOdometryTool(BaseTool):
    """Tool to get current robot position and velocity."""

    name: str = "GetOdometry"
    description: str = (
        "Get the current position and velocity of the robot. "
        "Returns x, y coordinates, orientation (yaw), and current velocities. "
        "Use this when you need to know where the robot is or how fast it's moving."
    )
    args_schema: Type[BaseModel] = EmptyInput
    return_direct: bool = False

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get odometry data."""
        start_time = time.time()

        interface = get_status_interface()
        odom = interface.get_odometry()

        if odom is None:
            result = "Odometry data unavailable"
            success = False
        else:
            pos = odom["position"]
            result = (
                f"Robot Position:\n"
                f"  X: {pos['x']:.3f} m\n"
                f"  Y: {pos['y']:.3f} m\n"
                f"  Orientation: {odom['orientation_yaw_deg']:.1f}°\n"
                f"  Linear velocity: {odom['linear_velocity']:.3f} m/s\n"
                f"  Angular velocity: {odom['angular_velocity']:.3f} rad/s"
            )
            success = True

        execution_time = (time.time() - start_time) * 1000
        log_tool_call(
            tool_name=self.name,
            parameters={},
            result=result,
            success=success,
            execution_time_ms=execution_time,
        )

        return result


class ListNodesTool(BaseTool):
    """Tool to list active ROS 2 nodes."""

    name: str = "ListNodes"
    description: str = (
        "List all active ROS 2 nodes in the system. "
        "Use this for diagnostics or to verify specific nodes are running."
    )
    args_schema: Type[BaseModel] = EmptyInput
    return_direct: bool = False

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """List active nodes."""
        start_time = time.time()

        interface = get_status_interface()
        nodes = interface.list_nodes()

        if not nodes:
            result = "No active nodes found (or unable to query)"
            success = False
        else:
            result = f"Active ROS 2 nodes ({len(nodes)}):\n"
            result += "\n".join(f"  - {node}" for node in nodes)
            success = True

        execution_time = (time.time() - start_time) * 1000
        log_tool_call(
            tool_name=self.name,
            parameters={},
            result=result,
            success=success,
            execution_time_ms=execution_time,
        )

        return result


class ListTopicsTool(BaseTool):
    """Tool to list active ROS 2 topics."""

    name: str = "ListTopics"
    description: str = (
        "List all active ROS 2 topics in the system. "
        "Use this for diagnostics or to see what data is being published."
    )
    args_schema: Type[BaseModel] = EmptyInput
    return_direct: bool = False

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """List active topics."""
        start_time = time.time()

        interface = get_status_interface()
        topics = interface.list_topics()

        if not topics:
            result = "No active topics found (or unable to query)"
            success = False
        else:
            result = f"Active ROS 2 topics ({len(topics)}):\n"
            for topic_name, topic_types in topics:
                type_str = ", ".join(topic_types)
                result += f"  - {topic_name} [{type_str}]\n"
            success = True

        execution_time = (time.time() - start_time) * 1000
        log_tool_call(
            tool_name=self.name,
            parameters={},
            result=result,
            success=success,
            execution_time_ms=execution_time,
        )

        return result


# Convenience function to create all status tools
def get_status_tools() -> list[BaseTool]:
    """Get all status-related tools."""
    return [
        BatteryStatusTool(),
        SystemHealthTool(),
        GetOdometryTool(),
        ListNodesTool(),
        ListTopicsTool(),
    ]
