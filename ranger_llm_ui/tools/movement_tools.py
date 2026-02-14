"""
Movement Tools - Robot movement control tools for the LangChain agent.

These tools wrap ROS 2 actions for robot movement:
- MoveForward: Move the robot forward by a specified distance
- MoveBackward: Move the robot backward by a specified distance
- TurnAngle: Rotate the robot in place by a specified angle
- StopRobot: Immediately stop the robot
"""

import time
import math
import logging
from typing import Optional, Type, Any

from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from ranger_llm_ui.safety.guard import get_safety_guard, SafetyGuard
from ranger_llm_ui.utils.logger import log_tool_call
from ranger_llm_ui.schemas.commands import (
    MoveCommand,
    TurnCommand,
    StopCommand,
    Direction,
    CommandStatus,
)

logger = logging.getLogger(__name__)

# Try to import ROS 2, but allow running without it for testing
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning("ROS 2 (rclpy) not available. Running in simulation mode.")


class ROSInterface:
    """
    Singleton interface for ROS 2 communication.

    This class manages the ROS 2 node and provides methods for publishing
    velocity commands and subscribing to odometry.
    """

    _instance: Optional["ROSInterface"] = None
    _node: Optional[Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, node: Optional[Any] = None):
        """Initialize the ROS interface with a node."""
        # Allow re-initialization when we previously started in simulation mode
        # and a real ROS node becomes available later.
        if self._initialized and not (self._simulation_mode and node is not None):
            return

        self._node = node
        self._cmd_vel_pub = None
        self._odom_sub = None
        self._current_odom: Optional[Any] = None
        self._simulation_mode = not ROS_AVAILABLE or node is None

        if not self._simulation_mode and self._node is not None:
            # Create publisher for velocity commands
            self._cmd_vel_pub = self._node.create_publisher(
                Twist, "/cmd_vel", 10
            )
            # Subscribe to odometry for position tracking
            self._odom_sub = self._node.create_subscription(
                Odometry,
                "/odom",
                self._odom_callback,
                10
            )
            logger.info("ROS interface initialized with node")
        else:
            logger.info("ROS interface running in simulation mode")

        self._initialized = True

    def _odom_callback(self, msg):
        """Callback for odometry messages."""
        self._current_odom = msg

    @property
    def simulation_mode(self) -> bool:
        return self._simulation_mode

    def publish_velocity(self, linear_x: float = 0.0, angular_z: float = 0.0):
        """Publish a velocity command."""
        if self._simulation_mode:
            logger.debug(f"[SIM] Publishing velocity: linear={linear_x}, angular={angular_z}")
            return

        if self._cmd_vel_pub is not None:
            msg = Twist()
            msg.linear.x = linear_x
            msg.angular.z = angular_z
            self._cmd_vel_pub.publish(msg)

    def stop(self):
        """Send stop command (zero velocity)."""
        self.publish_velocity(0.0, 0.0)
        # Publish multiple times to ensure delivery
        for _ in range(3):
            self.publish_velocity(0.0, 0.0)
            if not self._simulation_mode:
                time.sleep(0.05)

    def get_current_position(self) -> Optional[tuple[float, float, float]]:
        """Get current position (x, y, yaw) from odometry."""
        if self._simulation_mode:
            return (0.0, 0.0, 0.0)

        if self._current_odom is None:
            return None

        pos = self._current_odom.pose.pose.position
        ori = self._current_odom.pose.pose.orientation

        # Convert quaternion to yaw
        siny_cosp = 2 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1 - 2 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (pos.x, pos.y, yaw)


# Global ROS interface instance
_ros_interface: Optional[ROSInterface] = None


def get_ros_interface() -> ROSInterface:
    """Get or create the ROS interface singleton."""
    global _ros_interface
    if _ros_interface is None:
        _ros_interface = ROSInterface()
    return _ros_interface


def initialize_ros_interface(node: Optional[Any] = None):
    """Initialize the ROS interface with a node."""
    interface = get_ros_interface()
    interface.initialize(node)
    return interface


# Pydantic models for tool arguments
class MoveForwardInput(BaseModel):
    """Input schema for MoveForward tool."""
    distance_m: float = Field(
        description="Distance to move forward in meters (0.1 to 5.0)",
        ge=0.1,
        le=5.0
    )


class MoveBackwardInput(BaseModel):
    """Input schema for MoveBackward tool."""
    distance_m: float = Field(
        description="Distance to move backward in meters (0.1 to 5.0)",
        ge=0.1,
        le=5.0
    )


class TurnAngleInput(BaseModel):
    """Input schema for TurnAngle tool."""
    angle_deg: float = Field(
        description="Angle to turn in degrees. Positive = turn right/clockwise, Negative = turn left/counterclockwise",
        ge=-360,
        le=360
    )


class StopRobotInput(BaseModel):
    """Input schema for StopRobot tool."""
    reason: Optional[str] = Field(
        default=None,
        description="Optional reason for stopping"
    )


class MoveForwardTool(BaseTool):
    """Tool to move the robot forward by a specified distance."""

    name: str = "MoveForward"
    description: str = (
        "Move the robot forward by a specified distance in meters. "
        "Use this when you need to move the robot forward. "
        "Input should be the distance in meters (0.1 to 5.0)."
    )
    args_schema: Type[BaseModel] = MoveForwardInput
    return_direct: bool = False

    safety_guard: SafetyGuard = Field(default_factory=get_safety_guard)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        distance_m: float,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the move forward command."""
        start_time = time.time()

        # Safety check
        is_safe, msg, params = self.safety_guard.check_command_safety(
            "move", distance_m=distance_m
        )

        if not is_safe:
            requires_confirmation = "confirmation" in msg.lower()
            prefix = "CONFIRMATION REQUIRED" if requires_confirmation else "SAFETY BLOCKED"
            log_tool_call(
                tool_name=self.name,
                parameters={"distance_m": distance_m},
                result=msg,
                success=False,
                error="Confirmation required" if requires_confirmation else msg,
            )
            return f"{prefix}: {msg}"

        validated_distance = params.get("distance_m", distance_m)
        velocity = params.get("velocity_mps", 0.2)

        # Create command record
        command = MoveCommand(
            direction=Direction.FORWARD,
            distance_m=validated_distance,
            velocity_mps=velocity,
        )

        ros = get_ros_interface()

        if ros.simulation_mode:
            # Simulate movement
            travel_time = validated_distance / velocity
            logger.info(
                f"[SIM] Moving forward {validated_distance}m at {velocity}m/s "
                f"(would take {travel_time:.1f}s)"
            )
            time.sleep(min(travel_time, 1.0))  # Cap simulation time
            result = f"Moved forward {validated_distance:.2f} meters (simulated)"
        else:
            # Real robot movement
            try:
                travel_time = validated_distance / velocity
                start_pos = ros.get_current_position()

                # Publish velocity command
                ros.publish_velocity(linear_x=velocity)

                # Wait for movement to complete (simplified - real impl would use odometry)
                time.sleep(travel_time)

                # Stop
                ros.stop()

                command.status = CommandStatus.COMPLETED
                result = f"Moved forward {validated_distance:.2f} meters"

            except Exception as e:
                ros.stop()
                command.status = CommandStatus.FAILED
                command.error_message = str(e)
                log_tool_call(
                    tool_name=self.name,
                    parameters={"distance_m": distance_m},
                    result=None,
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
                return f"Error moving forward: {e}"

        execution_time = (time.time() - start_time) * 1000
        log_tool_call(
            tool_name=self.name,
            parameters={"distance_m": validated_distance},
            result=result,
            success=True,
            execution_time_ms=execution_time,
        )

        return result


class MoveBackwardTool(BaseTool):
    """Tool to move the robot backward by a specified distance."""

    name: str = "MoveBackward"
    description: str = (
        "Move the robot backward by a specified distance in meters. "
        "Use this when you need to move the robot backward or reverse. "
        "Input should be the distance in meters (0.1 to 5.0)."
    )
    args_schema: Type[BaseModel] = MoveBackwardInput
    return_direct: bool = False

    safety_guard: SafetyGuard = Field(default_factory=get_safety_guard)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        distance_m: float,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the move backward command."""
        start_time = time.time()

        # Safety check
        is_safe, msg, params = self.safety_guard.check_command_safety(
            "move", distance_m=distance_m
        )

        if not is_safe:
            requires_confirmation = "confirmation" in msg.lower()
            prefix = "CONFIRMATION REQUIRED" if requires_confirmation else "SAFETY BLOCKED"
            log_tool_call(
                tool_name=self.name,
                parameters={"distance_m": distance_m},
                result=msg,
                success=False,
                error="Confirmation required" if requires_confirmation else msg,
            )
            return f"{prefix}: {msg}"

        validated_distance = params.get("distance_m", distance_m)
        velocity = params.get("velocity_mps", 0.2)

        ros = get_ros_interface()

        if ros.simulation_mode:
            travel_time = validated_distance / velocity
            logger.info(
                f"[SIM] Moving backward {validated_distance}m at {velocity}m/s"
            )
            time.sleep(min(travel_time, 1.0))
            result = f"Moved backward {validated_distance:.2f} meters (simulated)"
        else:
            try:
                travel_time = validated_distance / velocity
                ros.publish_velocity(linear_x=-velocity)
                time.sleep(travel_time)
                ros.stop()
                result = f"Moved backward {validated_distance:.2f} meters"
            except Exception as e:
                ros.stop()
                log_tool_call(
                    tool_name=self.name,
                    parameters={"distance_m": distance_m},
                    result=None,
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
                return f"Error moving backward: {e}"

        execution_time = (time.time() - start_time) * 1000
        log_tool_call(
            tool_name=self.name,
            parameters={"distance_m": validated_distance},
            result=result,
            success=True,
            execution_time_ms=execution_time,
        )

        return result


class TurnAngleTool(BaseTool):
    """Tool to rotate the robot in place by a specified angle."""

    name: str = "TurnAngle"
    description: str = (
        "Rotate the robot in place by a specified angle in degrees. "
        "Positive angle = turn right/clockwise. "
        "Negative angle = turn left/counterclockwise. "
        "Input should be the angle in degrees (-360 to 360)."
    )
    args_schema: Type[BaseModel] = TurnAngleInput
    return_direct: bool = False

    safety_guard: SafetyGuard = Field(default_factory=get_safety_guard)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        angle_deg: float,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the turn command."""
        start_time = time.time()

        # Safety check
        is_safe, msg, params = self.safety_guard.check_command_safety(
            "turn", angle_deg=angle_deg
        )

        if not is_safe:
            requires_confirmation = "confirmation" in msg.lower()
            prefix = "CONFIRMATION REQUIRED" if requires_confirmation else "SAFETY BLOCKED"
            log_tool_call(
                tool_name=self.name,
                parameters={"angle_deg": angle_deg},
                result=msg,
                success=False,
                error="Confirmation required" if requires_confirmation else msg,
            )
            return f"{prefix}: {msg}"

        validated_angle = params.get("angle_deg", angle_deg)
        angular_velocity_dps = params.get("angular_velocity_dps", 30.0)  # deg/s
        angular_velocity_rad = angular_velocity_dps * (math.pi / 180)

        ros = get_ros_interface()

        if ros.simulation_mode:
            turn_time = abs(validated_angle) / angular_velocity_dps
            direction = "right" if validated_angle > 0 else "left"
            logger.info(
                f"[SIM] Turning {direction} {abs(validated_angle)}° "
                f"at {angular_velocity_dps}°/s"
            )
            time.sleep(min(turn_time, 1.0))
            result = f"Turned {direction} {abs(validated_angle):.1f} degrees (simulated)"
        else:
            try:
                turn_time = abs(validated_angle) / angular_velocity_dps
                # Positive angle = clockwise = negative angular.z in ROS convention
                angular_z = -angular_velocity_rad if validated_angle > 0 else angular_velocity_rad

                ros.publish_velocity(angular_z=angular_z)
                time.sleep(turn_time)
                ros.stop()

                direction = "right" if validated_angle > 0 else "left"
                result = f"Turned {direction} {abs(validated_angle):.1f} degrees"
            except Exception as e:
                ros.stop()
                log_tool_call(
                    tool_name=self.name,
                    parameters={"angle_deg": angle_deg},
                    result=None,
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
                return f"Error turning: {e}"

        execution_time = (time.time() - start_time) * 1000
        log_tool_call(
            tool_name=self.name,
            parameters={"angle_deg": validated_angle},
            result=result,
            success=True,
            execution_time_ms=execution_time,
        )

        return result


class StopRobotTool(BaseTool):
    """Tool to immediately stop the robot."""

    name: str = "StopRobot"
    description: str = (
        "Immediately stop all robot movement. "
        "Use this in emergencies or when you need to halt the robot. "
        "This will cancel any ongoing movement commands."
    )
    args_schema: Type[BaseModel] = StopRobotInput
    return_direct: bool = False

    def _run(
        self,
        reason: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the stop command."""
        start_time = time.time()

        ros = get_ros_interface()

        if ros.simulation_mode:
            logger.info(f"[SIM] Robot stopped. Reason: {reason or 'User request'}")
            result = "Robot stopped (simulated)"
        else:
            try:
                ros.stop()
                result = "Robot stopped"
            except Exception as e:
                log_tool_call(
                    tool_name=self.name,
                    parameters={"reason": reason},
                    result=None,
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
                return f"Error stopping robot: {e}"

        if reason:
            result += f". Reason: {reason}"

        execution_time = (time.time() - start_time) * 1000
        log_tool_call(
            tool_name=self.name,
            parameters={"reason": reason},
            result=result,
            success=True,
            execution_time_ms=execution_time,
        )

        return result


# Convenience function to create all movement tools
def get_movement_tools() -> list[BaseTool]:
    """Get all movement-related tools."""
    return [
        MoveForwardTool(),
        MoveBackwardTool(),
        TurnAngleTool(),
        StopRobotTool(),
    ]
