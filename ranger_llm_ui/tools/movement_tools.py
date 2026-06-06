"""
Movement Tools - Robot movement control tools for the LangChain agent.

These tools wrap ROS 2 actions for robot movement:
- MoveForward: Move the robot forward by a specified distance
- MoveBackward: Move the robot backward by a specified distance
- TurnAngle: Rotate the robot in place by a specified angle
- StopRobot: Immediately stop the robot
"""

import os
import time
import math
import logging
import threading
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
    from rclpy.action import ActionClient
    from geometry_msgs.msg import Twist, PoseStamped
    from nav_msgs.msg import Odometry
    from action_msgs.msg import GoalStatus
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning("ROS 2 (rclpy) not available. Running in simulation mode.")

# Try to import the custom dead-reckoning action definitions. These back the
# relative movement primitives (MoveForward/MoveBackward/TurnAngle) via the
# closed-loop odometry action server in movement_action_server.py.
try:
    from ranger_llm_msgs.action import DriveDistance, RotateAngle
    ACTIONS_AVAILABLE = True
except ImportError:
    ACTIONS_AVAILABLE = False
    if ROS_AVAILABLE:
        logger.warning(
            "ranger_llm_msgs not available. "
            "Build ranger_llm_msgs package first: colcon build --packages-select ranger_llm_msgs"
        )

# Try to import the Nav2 NavigateToPose action. This backs the NavigateToPose
# tool — the obstacle-aware autonomous navigation ("2D Nav Goal"). It is
# independent of ranger_llm_msgs and only needs nav2_msgs + geometry_msgs.
try:
    from nav2_msgs.action import NavigateToPose
    NAV2_AVAILABLE = True
except ImportError:
    NAV2_AVAILABLE = False
    if ROS_AVAILABLE:
        logger.warning(
            "nav2_msgs not available; the NavigateToPose tool will report an "
            "error until a Nav2 install is sourced. Relative moves "
            "(MoveForward/MoveBackward/TurnAngle) are unaffected."
        )

# Try to import std_srvs/Trigger. This backs the ZeroOdometry tool, which calls
# the robot-side odom-reset relay service (std_srvs/srv/Trigger) to re-zero the
# /odom topic at its source. std_srvs ships with a standard ROS 2 install.
try:
    from std_srvs.srv import Trigger
    STD_SRVS_AVAILABLE = True
except ImportError:
    STD_SRVS_AVAILABLE = False
    if ROS_AVAILABLE:
        logger.warning(
            "std_srvs not available; the ZeroOdometry tool will report an error "
            "until std_srvs is sourced."
        )


# Default timeouts
ACTION_SERVER_WAIT_TIMEOUT_S = 5.0
ACTION_GOAL_TIMEOUT_S = 60.0

# ZeroOdometry: name of the std_srvs/Trigger service exposed by the robot-side
# odom_reset_relay node (env-overridable via RESET_ODOM_SERVICE).
RESET_ODOM_SERVICE = os.getenv("RESET_ODOM_SERVICE", "/reset_odom")

# Nav2 NavigateToPose ("2D Nav Goal") configuration (env-overridable).
# NAV2_NAVIGATE_ACTION - action server name (default: navigate_to_pose).
# NAV2_GOAL_FRAME      - frame the goal pose is expressed in (default: map, the
#                        Nav2 global frame).
# NAV2_GOAL_TIMEOUT_S  - max client-side wait for Nav2 to reach the goal. Nav2
#                        plans around obstacles, so journeys can be long.
NAV2_ACTION_NAME = os.getenv("NAV2_NAVIGATE_ACTION", "navigate_to_pose")
NAV2_GOAL_FRAME = os.getenv("NAV2_GOAL_FRAME", "map")
NAV2_GOAL_TIMEOUT_S = float(os.getenv("NAV2_GOAL_TIMEOUT_S", "300.0"))

# MoveToPose (turn-drive-turn) tolerances. A sub-step whose magnitude is below
# the relevant threshold is skipped as a no-op (e.g. a near-zero offset becomes
# a pure rotation; a near-zero heading correction is skipped).
POSITION_EPS_M = 0.02              # 2 cm — below this there is no translation
HEADING_EPS_RAD = math.radians(0.5)  # 0.5° — below this there is no rotation


def _normalize_angle_rad(angle_rad: float) -> float:
    """Normalize an angle to the range [-pi, pi] (shortest equivalent turn)."""
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


class ROSInterface:
    """
    Singleton interface for ROS 2 communication.

    This class manages the ROS 2 node and provides methods for sending
    movement action goals and subscribing to odometry.
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
        self._drive_client = None
        self._rotate_client = None
        self._nav_to_pose_client = None
        self._reset_odom_client = None
        self._active_goal_handle = None
        self._goal_lock = threading.Lock()

        if not self._simulation_mode and self._node is not None:
            # Create publisher for emergency stop (direct velocity override)
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

            # Create dead-reckoning action clients (relative move primitives)
            if ACTIONS_AVAILABLE:
                self._drive_client = ActionClient(
                    self._node, DriveDistance, 'drive_distance'
                )
                self._rotate_client = ActionClient(
                    self._node, RotateAngle, 'rotate_angle'
                )
                logger.info("ROS interface initialized with action clients")
            else:
                logger.warning(
                    "ROS interface initialized WITHOUT dead-reckoning action "
                    "clients (ranger_llm_msgs not available)"
                )

            # Create the Nav2 NavigateToPose client (autonomous navigation).
            # Independent of ranger_llm_msgs.
            if NAV2_AVAILABLE:
                self._nav_to_pose_client = ActionClient(
                    self._node, NavigateToPose, NAV2_ACTION_NAME
                )
                logger.info(
                    f"Nav2 NavigateToPose client created ('{NAV2_ACTION_NAME}')"
                )
            else:
                logger.warning(
                    "Nav2 NavigateToPose client NOT created (nav2_msgs "
                    "unavailable); NavigateToPose tool will report an error"
                )

            # Create the odom-reset service client (ZeroOdometry).
            if STD_SRVS_AVAILABLE:
                self._reset_odom_client = self._node.create_client(
                    Trigger, RESET_ODOM_SERVICE
                )
                logger.info(
                    f"ZeroOdometry client created ('{RESET_ODOM_SERVICE}')"
                )
            else:
                logger.warning(
                    "ZeroOdometry client NOT created (std_srvs unavailable); "
                    "ZeroOdometry tool will report an error"
                )
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
        """Publish a velocity command directly (used for emergency stop)."""
        if self._simulation_mode:
            logger.debug(f"[SIM] Publishing velocity: linear={linear_x}, angular={angular_z}")
            return

        if self._cmd_vel_pub is not None:
            msg = Twist()
            msg.linear.x = linear_x
            msg.angular.z = angular_z
            self._cmd_vel_pub.publish(msg)

    def stop(self):
        """Cancel active goals and send stop command (zero velocity)."""
        self.cancel_active_goal()
        self.publish_velocity(0.0, 0.0)
        # Publish multiple times to ensure delivery
        for _ in range(3):
            self.publish_velocity(0.0, 0.0)
            if not self._simulation_mode:
                time.sleep(0.05)

    def cancel_active_goal(self):
        """Cancel the currently active action goal, if any."""
        with self._goal_lock:
            if self._active_goal_handle is not None:
                logger.info("Cancelling active movement goal")
                try:
                    self._active_goal_handle.cancel_goal_async()
                except Exception as e:
                    logger.warning(f"Error cancelling goal: {e}")
                self._active_goal_handle = None

    @staticmethod
    def _wait_for_future(future, timeout_s: float) -> bool:
        """
        Wait for an rclpy future to complete using a threading Event.

        This avoids calling rclpy.spin_until_future_complete() which deadlocks
        when the node is already being spun by a MultiThreadedExecutor.

        Returns True if the future completed, False on timeout.
        """
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        return event.wait(timeout=timeout_s)

    def drive_distance(
        self,
        distance_m: float,
        max_velocity_mps: float,
        timeout_s: float = ACTION_GOAL_TIMEOUT_S,
    ) -> tuple[bool, str]:
        """
        Send a DriveDistance action goal and block until completion.

        Args:
            distance_m: Distance in meters (positive=forward, negative=backward).
            max_velocity_mps: Maximum velocity in m/s.
            timeout_s: Maximum time to wait for result.

        Returns:
            (success, message): ``success`` is True only when the action server
            reported the goal completed (not cancelled/aborted); ``message`` is
            a human-readable result string.
        """
        if self._drive_client is None:
            return False, "Error: DriveDistance action client not available"

        # Wait for action server
        if not self._drive_client.wait_for_server(
            timeout_sec=ACTION_SERVER_WAIT_TIMEOUT_S
        ):
            return (
                False,
                "Error: drive_distance action server not available. "
                "Ensure the movement_action_server node is running.",
            )

        # Build goal
        goal_msg = DriveDistance.Goal()
        goal_msg.distance_m = float(distance_m)
        goal_msg.max_velocity_mps = float(max_velocity_mps)

        logger.info(f"Sending drive goal: {distance_m:.2f}m at max {max_velocity_mps:.2f} m/s")

        # Send goal (future is resolved by the executor's background thread)
        send_future = self._drive_client.send_goal_async(goal_msg)
        if not self._wait_for_future(send_future, timeout_s=5.0):
            return False, "Error: Timed out waiting for drive goal to be accepted"

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, "Error: Drive goal was rejected by action server"

        # Track active goal for cancellation
        with self._goal_lock:
            self._active_goal_handle = goal_handle

        # Wait for result
        result_future = goal_handle.get_result_async()
        completed = self._wait_for_future(result_future, timeout_s=timeout_s)

        # Clear active goal
        with self._goal_lock:
            self._active_goal_handle = None

        if not completed or result_future.result() is None:
            self.publish_velocity(0.0, 0.0)
            return False, f"Error: Drive action timed out after {timeout_s:.0f}s"

        result = result_future.result().result
        return bool(result.success), result.message

    def rotate_angle(
        self,
        angle_deg: float,
        max_angular_velocity_dps: float,
        timeout_s: float = ACTION_GOAL_TIMEOUT_S,
    ) -> tuple[bool, str]:
        """
        Send a RotateAngle action goal and block until completion.

        Args:
            angle_deg: Angle in degrees (positive=clockwise, negative=CCW).
            max_angular_velocity_dps: Maximum angular velocity in degrees/second.
            timeout_s: Maximum time to wait for result.

        Returns:
            (success, message): ``success`` is True only when the action server
            reported the goal completed (not cancelled/aborted); ``message`` is
            a human-readable result string.
        """
        if self._rotate_client is None:
            return False, "Error: RotateAngle action client not available"

        # Wait for action server
        if not self._rotate_client.wait_for_server(
            timeout_sec=ACTION_SERVER_WAIT_TIMEOUT_S
        ):
            return (
                False,
                "Error: rotate_angle action server not available. "
                "Ensure the movement_action_server node is running.",
            )

        # Build goal
        goal_msg = RotateAngle.Goal()
        goal_msg.angle_deg = float(angle_deg)
        goal_msg.max_angular_velocity_dps = float(max_angular_velocity_dps)

        logger.info(
            f"Sending rotate goal: {angle_deg:.1f}° at max "
            f"{max_angular_velocity_dps:.1f} deg/s"
        )

        # Send goal
        send_future = self._rotate_client.send_goal_async(goal_msg)
        if not self._wait_for_future(send_future, timeout_s=5.0):
            return False, "Error: Timed out waiting for rotate goal to be accepted"

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, "Error: Rotate goal was rejected by action server"

        # Track active goal for cancellation
        with self._goal_lock:
            self._active_goal_handle = goal_handle

        # Wait for result
        result_future = goal_handle.get_result_async()
        completed = self._wait_for_future(result_future, timeout_s=timeout_s)

        # Clear active goal
        with self._goal_lock:
            self._active_goal_handle = None

        if not completed or result_future.result() is None:
            self.publish_velocity(0.0, 0.0)
            return False, f"Error: Rotate action timed out after {timeout_s:.0f}s"

        result = result_future.result().result
        return bool(result.success), result.message

    @staticmethod
    def _describe_turn(angle_rad: float) -> str:
        """Human-readable description of a (CCW-positive) turn."""
        direction = "left" if angle_rad > 0 else "right"
        return f"{abs(math.degrees(angle_rad)):.1f}° {direction}"

    def move_to_pose(
        self,
        x: float,
        y: float,
        yaw_rad: Optional[float] = None,
        max_velocity_mps: float = 0.2,
        max_angular_velocity_dps: float = 30.0,
        timeout_s: float = ACTION_GOAL_TIMEOUT_S,
        current_pose: Optional[tuple[float, float, float]] = None,
    ) -> str:
        """
        Drive to an ABSOLUTE goal pose in the odometry frame using a
        dead-reckoning turn-drive-turn maneuver.

        The goal (x, y, yaw_rad) is expressed in the absolute odometry frame —
        the one reset by ``ZeroOdometry`` (+x, +y, yaw CCW-positive, REP-103).
        The maneuver is planned from the robot's CURRENT pose in that frame:
          1. rotate in place to FACE the goal point,
          2. drive straight to it (``hypot`` of the offset),
          3. if ``yaw_rad`` is given, rotate to the requested FINAL heading.

        ``current_pose`` lets the caller pass the pose it already read so the
        safety estimate and the executed plan stay consistent; if None it is
        read here via :meth:`get_current_position`. A goal already at the
        current position collapses to a pure rotation to ``yaw_rad``; a
        sub-rotation below HEADING_EPS_RAD is skipped.

        This is pure dead-reckoning with NO obstacle avoidance — use
        ``navigate_to_pose`` (Nav2) for absolute, obstacle-aware goals.

        Returns a human-readable summary, or an early "Move-to-pose aborted ..."
        message if a sub-step fails / is cancelled (e.g. via StopRobot).
        """
        if current_pose is None:
            current_pose = self.get_current_position()
        if current_pose is None:
            return (
                "Error: no odometry available (/odom) — cannot drive to an "
                "absolute pose. Ensure the base driver is publishing /odom."
            )

        cx, cy, cyaw = current_pose
        dx = x - cx
        dy = y - cy
        distance = math.hypot(dx, dy)
        translating = distance >= POSITION_EPS_M
        # Absolute bearing from the robot to the goal point, or keep the current
        # heading as the reference when there is no translation.
        bearing = math.atan2(dy, dx) if translating else cyaw

        steps: list[str] = []

        # ── Phase 1: rotate to face the destination point ──────────────────
        if translating:
            turn1 = _normalize_angle_rad(bearing - cyaw)
            if abs(turn1) >= HEADING_EPS_RAD:
                label = self._describe_turn(turn1)
                if self._simulation_mode:
                    time.sleep(0.2)
                    ok = True
                else:
                    # rotate_angle uses positive=clockwise, so negate the
                    # CCW-positive turn.
                    ok, msg = self.rotate_angle(
                        -math.degrees(turn1), max_angular_velocity_dps, timeout_s
                    )
                if not ok:
                    return (
                        "Move-to-pose aborted while turning to face the "
                        f"destination: {msg}"
                    )
                steps.append(f"faced the destination (turned {label})")

        # ── Phase 2: drive straight to the destination ─────────────────────
        if translating:
            if self._simulation_mode:
                time.sleep(min(distance / max(max_velocity_mps, 1e-3), 1.0))
                ok = True
            else:
                ok, msg = self.drive_distance(distance, max_velocity_mps, timeout_s)
            if not ok:
                return f"Move-to-pose aborted while driving to the destination: {msg}"
            steps.append(f"drove {distance:.2f} m")

        # ── Phase 3: rotate to the final absolute heading ──────────────────
        if yaw_rad is not None:
            # After phases 1-2 the robot points along ``bearing`` (or kept
            # ``cyaw`` when not translating). Turn the shortest way to yaw_rad.
            remaining = _normalize_angle_rad(yaw_rad - bearing)
            if abs(remaining) >= HEADING_EPS_RAD:
                label = self._describe_turn(remaining)
                if self._simulation_mode:
                    time.sleep(0.2)
                    ok = True
                else:
                    ok, msg = self.rotate_angle(
                        -math.degrees(remaining), max_angular_velocity_dps, timeout_s
                    )
                if not ok:
                    return (
                        "Move-to-pose aborted while turning to the final "
                        f"heading: {msg}"
                    )
                steps.append(f"turned to the final heading (turned {label})")

        if not steps:
            return "Already at the requested pose; no movement needed."

        summary = (
            f"Reached pose ({x:.2f}, {y:.2f}"
            + (f", yaw={math.degrees(yaw_rad):.1f}°" if yaw_rad is not None else "")
            + "): "
            + ", then ".join(steps)
            + "."
        )
        if self._simulation_mode:
            summary += " (simulated)"
        return summary

    @staticmethod
    def _yaw_to_quaternion(yaw_rad: float) -> tuple[float, float]:
        """Return (z, w) of a unit quaternion for a yaw rotation about +Z.

        ROS convention: yaw is CCW-positive about +Z. x and y are always 0 for a
        pure-yaw (planar) rotation.
        """
        return math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)

    def navigate_to_pose(
        self,
        x: float,
        y: float,
        yaw_rad: Optional[float] = None,
        frame_id: str = NAV2_GOAL_FRAME,
        timeout_s: float = NAV2_GOAL_TIMEOUT_S,
    ) -> str:
        """
        Send a Nav2 NavigateToPose goal (the "2D Nav Goal") and block until done.

        Nav2 plans an obstacle-aware path to the goal and drives there in closed
        loop using its own costmaps and velocity limits — this is NOT
        dead-reckoning. The goal is tracked as the active goal handle so that
        StopRobot / stop() cancels the navigation.

        Args:
            x, y: Target position in ``frame_id`` (meters).
            yaw_rad: Final heading (radians, CCW-positive). None -> yaw 0 (the
                goal frame's +X axis).
            frame_id: TF frame the goal is expressed in (default: map).
            timeout_s: Max time to wait for the result.

        Returns:
            Human-readable result string.
        """
        if self._nav_to_pose_client is None:
            return (
                "Error: Nav2 NavigateToPose client not available. Ensure "
                "nav2_msgs is sourced and the Nav2 stack is running "
                "(e.g. robofi_bringup navigation)."
            )

        # Wait for the Nav2 action server
        if not self._nav_to_pose_client.wait_for_server(
            timeout_sec=ACTION_SERVER_WAIT_TIMEOUT_S
        ):
            return (
                f"Error: Nav2 '{NAV2_ACTION_NAME}' action server not available. "
                "Ensure the Nav2 stack is running (robofi_bringup navigation)."
            )

        # Build the PoseStamped goal
        goal_msg = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        # Leave stamp at 0 so Nav2 uses the latest available transform.
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        qz, qw = self._yaw_to_quaternion(yaw_rad if yaw_rad is not None else 0.0)
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        goal_msg.pose = pose

        yaw_str = (
            f"{math.degrees(yaw_rad):.1f}°" if yaw_rad is not None else "0° (default)"
        )
        logger.info(
            f"Sending Nav2 goal: ({x:.2f}, {y:.2f}) yaw={yaw_str} in '{frame_id}'"
        )

        # Track the latest feedback (distance remaining) for richer reporting.
        feedback_state = {"distance_remaining": None}

        def _feedback_cb(fb_msg):
            try:
                feedback_state["distance_remaining"] = fb_msg.feedback.distance_remaining
            except Exception:
                pass

        # Send goal (future resolved by the executor's background thread)
        send_future = self._nav_to_pose_client.send_goal_async(
            goal_msg, feedback_callback=_feedback_cb
        )
        if not self._wait_for_future(send_future, timeout_s=5.0):
            return "Error: Timed out waiting for Nav2 goal to be accepted"

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return "Error: Nav2 goal was rejected by the action server"

        # Track active goal so StopRobot can cancel the navigation.
        with self._goal_lock:
            self._active_goal_handle = goal_handle

        # Wait for result
        result_future = goal_handle.get_result_async()
        completed = self._wait_for_future(result_future, timeout_s=timeout_s)

        # Clear active goal
        with self._goal_lock:
            self._active_goal_handle = None

        if not completed or result_future.result() is None:
            # Cancel so the robot stops if we gave up waiting.
            try:
                goal_handle.cancel_goal_async()
            except Exception as e:
                logger.warning(f"Error cancelling timed-out Nav2 goal: {e}")
            return f"Error: Nav2 navigation timed out after {timeout_s:.0f}s"

        status = result_future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            return (
                f"Reached goal ({x:.2f}, {y:.2f}, yaw={yaw_str}) in frame "
                f"'{frame_id}' via Nav2."
            )
        if status == GoalStatus.STATUS_CANCELED:
            return f"Navigation to ({x:.2f}, {y:.2f}) was cancelled."
        if status == GoalStatus.STATUS_ABORTED:
            dr = feedback_state["distance_remaining"]
            extra = (
                f" ({dr:.2f} m still remaining)"
                if isinstance(dr, (int, float)) and dr is not None
                else ""
            )
            return (
                f"Navigation to ({x:.2f}, {y:.2f}) was aborted by Nav2{extra}. "
                "The planner could not reach the goal — it may be blocked, "
                "off-map, or unreachable. Check the goal and the costmap."
            )
        return (
            f"Navigation to ({x:.2f}, {y:.2f}) finished with status code {status}."
        )

    def get_current_position(self) -> Optional[tuple[float, float, float]]:
        """
        Get the current pose (x, y, yaw) from the /odom topic.

        The odom frame is re-zeroed at its source by the robot-side odom-reset
        relay (see :meth:`zero_odometry`), so after a ZeroOdometry call /odom —
        and therefore this pose — reads ~(0, 0, 0) at the robot's location.
        Returns None if no odometry has been received yet.
        """
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

    def zero_odometry(self) -> str:
        """
        Re-zero odometry by calling the robot-side relay service.

        Calls the ``RESET_ODOM_SERVICE`` (``std_srvs/srv/Trigger``, default
        ``/reset_odom``) exposed by the ``odom_reset_relay`` node, which snaps
        the /odom topic to (0, 0, 0) at the robot's current pose. Because the
        reset happens at the source, every /odom subscriber (including the
        GetOdometry readout) sees the zeroed frame. It does NOT reset any
        map/AMCL/FASTLIO localization.

        Returns the service's human-readable message (or an error string).
        """
        if self._simulation_mode:
            return "Odometry reset to (0, 0, 0) (simulated)."

        if self._reset_odom_client is None:
            return (
                "Error: ZeroOdometry service client not available (std_srvs not "
                "sourced or ROS node not initialized)."
            )

        if not self._reset_odom_client.wait_for_service(
            timeout_sec=ACTION_SERVER_WAIT_TIMEOUT_S
        ):
            return (
                f"Error: '{RESET_ODOM_SERVICE}' service not available. Ensure the "
                "odom_reset_relay node is running (ros2 node list | grep "
                "odom_reset_relay)."
            )

        future = self._reset_odom_client.call_async(Trigger.Request())
        if not self._wait_for_future(future, timeout_s=5.0):
            return f"Error: timed out calling '{RESET_ODOM_SERVICE}'."

        response = future.result()
        if response is None:
            return f"Error: no response from '{RESET_ODOM_SERVICE}'."

        if response.success:
            return (
                f"Odometry reset: {response.message}"
                if response.message
                else "Odometry reset: /odom is now (0, 0, 0) at the current pose."
            )
        return (
            f"Odometry reset failed: {response.message or 'service reported failure'}"
        )


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


class MoveToPoseInput(BaseModel):
    """Input schema for MoveToPose tool (absolute, odometry frame)."""
    x: float = Field(
        description=(
            "Absolute target X in the odometry frame, in meters (the frame "
            "reset by ZeroOdometry; +x is the forward direction at the last "
            "zero)."
        ),
        ge=-100.0,
        le=100.0,
    )
    y: float = Field(
        description=(
            "Absolute target Y in the odometry frame, in meters (+y is to the "
            "left of the forward direction at the last zero)."
        ),
        ge=-100.0,
        le=100.0,
    )
    yaw_deg: Optional[float] = Field(
        default=None,
        description=(
            "Optional FINAL absolute heading in the odometry frame, in degrees "
            "(CCW positive; 0 = the frame's +x axis). If omitted, the robot "
            "ends facing the destination (its direction of travel)."
        ),
        ge=-360,
        le=360,
    )


class ZeroOdometryInput(BaseModel):
    """Input schema for ZeroOdometry tool (no parameters)."""
    pass


class StopRobotInput(BaseModel):
    """Input schema for StopRobot tool."""
    reason: Optional[str] = Field(
        default=None,
        description="Optional reason for stopping"
    )


class NavigateToPoseInput(BaseModel):
    """Input schema for NavigateToPose tool."""
    x: float = Field(
        description="Target X position in the map frame (meters)"
    )
    y: float = Field(
        description="Target Y position in the map frame (meters)"
    )
    yaw_deg: Optional[float] = Field(
        default=None,
        description=(
            "Optional final heading in degrees (0=+X, 90=+Y, CCW positive). "
            "If omitted, the robot arrives at yaw 0 (facing the goal frame's "
            "+X axis)."
        ),
        ge=-360,
        le=360,
    )
    frame: Optional[str] = Field(
        default=None,
        description=(
            "TF frame the (x, y, yaw) goal is expressed in. Defaults to 'map' "
            "(the Nav2 global frame). Only override if the goal is given in "
            "another frame (e.g. 'odom')."
        ),
    )


class MoveForwardTool(BaseTool):
    """Tool to move the robot forward by a specified distance."""

    name: str = "MoveForward"
    description: str = (
        "LOW-LEVEL PRIMITIVE. Move the robot forward by a specified distance "
        "in meters along its current heading. Use ONLY for explicit relative "
        "commands like 'go forward 2 m'. For ANY absolute-position goal "
        "('go to (x,y)', 'return to origin'), use NavigateToPose instead. "
        "Input: distance in meters (0.1 to 5.0)."
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
        move_ok = True

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
            # Send action goal for closed-loop movement
            try:
                move_ok, result = ros.drive_distance(
                    distance_m=validated_distance,
                    max_velocity_mps=velocity,
                )
                command.status = (
                    CommandStatus.COMPLETED if move_ok else CommandStatus.FAILED
                )
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
            success=move_ok,
            execution_time_ms=execution_time,
        )

        return result


class MoveBackwardTool(BaseTool):
    """Tool to move the robot backward by a specified distance."""

    name: str = "MoveBackward"
    description: str = (
        "LOW-LEVEL PRIMITIVE. Move the robot backward by a specified distance "
        "in meters. Use ONLY for explicit relative commands like 'back up 1 m'. "
        "For 'return to origin' or any absolute goal, use NavigateToPose. "
        "Input: distance in meters (0.1 to 5.0)."
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
        move_ok = True

        if ros.simulation_mode:
            travel_time = validated_distance / velocity
            logger.info(
                f"[SIM] Moving backward {validated_distance}m at {velocity}m/s"
            )
            time.sleep(min(travel_time, 1.0))
            result = f"Moved backward {validated_distance:.2f} meters (simulated)"
        else:
            # Send action goal with negative distance for backward movement
            try:
                move_ok, result = ros.drive_distance(
                    distance_m=-validated_distance,
                    max_velocity_mps=velocity,
                )
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
            success=move_ok,
            execution_time_ms=execution_time,
        )

        return result


class TurnAngleTool(BaseTool):
    """Tool to rotate the robot in place by a specified angle."""

    name: str = "TurnAngle"
    description: str = (
        "LOW-LEVEL PRIMITIVE. Rotate in place by a specified angle in degrees. "
        "Positive=clockwise/right, negative=CCW/left. Use ONLY for explicit "
        "relative commands like 'turn left 90°'. For reaching an absolute "
        "(x,y,yaw) pose, use NavigateToPose instead — do NOT chain TurnAngle "
        "with MoveForward to compute absolute goals. Input: -360 to 360."
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

        ros = get_ros_interface()
        turn_ok = True

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
            # Send action goal for closed-loop rotation
            try:
                turn_ok, result = ros.rotate_angle(
                    angle_deg=validated_angle,
                    max_angular_velocity_dps=angular_velocity_dps,
                )
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
            success=turn_ok,
            execution_time_ms=execution_time,
        )

        return result


class MoveToPoseTool(BaseTool):
    """
    Drive to an ABSOLUTE pose in the odometry frame, via turn-drive-turn.

    A low-level dead-reckoning primitive that composes the closed-loop
    drive_distance / rotate_angle actions: rotate to face the destination
    point, drive straight to it, then rotate to the requested final heading.
    The goal (x, y, yaw_deg) is an absolute pose in the odometry frame — the one
    reset by ZeroOdometry (+x, +y, yaw CCW-positive). The maneuver is planned
    from the robot's current odom pose. There is NO obstacle avoidance — use
    NavigateToPose (Nav2) for absolute, obstacle-aware (map-frame) goals.
    """

    name: str = "MoveToPose"
    description: str = (
        "LOW-LEVEL PRIMITIVE. Drive to an ABSOLUTE pose in the odometry frame "
        "(the frame reset by ZeroOdometry) using dead-reckoning turn-drive-turn: "
        "(1) rotate in place to face the destination point, (2) drive straight to "
        "it, (3) rotate to the requested final heading. NO obstacle avoidance and "
        "NO map/localization — this is open-loop relative to wherever odometry was "
        "last zeroed, so prefer calling ZeroOdometry first. For obstacle-aware "
        "map-frame navigation use NavigateToPose instead. Inputs: x (m), y (m) "
        "absolute in the odom frame, optional yaw_deg (final absolute heading, CCW "
        "positive, 0=+x axis; omit to end facing the destination). Example: after "
        "ZeroOdometry, 'go 2 m ahead and 1 m left' -> x=2, y=1."
    )
    args_schema: Type[BaseModel] = MoveToPoseInput
    return_direct: bool = False

    safety_guard: SafetyGuard = Field(default_factory=get_safety_guard)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        x: float,
        y: float,
        yaw_deg: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the absolute move-to-pose (turn-drive-turn) command."""
        start_time = time.time()
        params = {"x": x, "y": y, "yaw_deg": yaw_deg}

        ros = get_ros_interface()

        # Plan from the robot's current pose in the (zeroable) odom frame. The
        # travel distance for the safety gate is the offset to the goal, so we
        # need the current pose up front.
        current_pose = ros.get_current_position()
        if current_pose is None:
            msg = (
                "Error: no odometry available (/odom) — cannot drive to an "
                "absolute pose. Ensure the base driver is publishing /odom."
            )
            log_tool_call(
                tool_name=self.name,
                parameters=params,
                result=msg,
                success=False,
                error=msg,
            )
            return msg

        distance = math.hypot(x - current_pose[0], y - current_pose[1])

        # Safety gate on the straight-line travel distance. check_command_safety
        # also enforces the operate gate (emergency stop / critical battery) and
        # the large-move confirmation threshold, mirroring MoveForward.
        is_safe, msg, move_params = self.safety_guard.check_command_safety(
            "move", distance_m=distance
        )
        if not is_safe:
            requires_confirmation = "confirmation" in msg.lower()
            prefix = (
                "CONFIRMATION REQUIRED" if requires_confirmation else "SAFETY BLOCKED"
            )
            log_tool_call(
                tool_name=self.name,
                parameters=params,
                result=msg,
                success=False,
                error="Confirmation required" if requires_confirmation else msg,
            )
            return f"{prefix}: {msg}"

        velocity = move_params.get("velocity_mps", 0.2)
        # Angular velocity is independent of the angle; pull the clamped value
        # from a turn safety check.
        _, _, turn_params = self.safety_guard.check_command_safety(
            "turn", angle_deg=0.0
        )
        angular_velocity_dps = turn_params.get("angular_velocity_dps", 30.0)

        yaw_rad = math.radians(yaw_deg) if yaw_deg is not None else None

        try:
            result = ros.move_to_pose(
                x=x,
                y=y,
                yaw_rad=yaw_rad,
                max_velocity_mps=velocity,
                max_angular_velocity_dps=angular_velocity_dps,
                current_pose=current_pose,
            )
            success = not (
                result.lower().startswith("move-to-pose aborted")
                or result.startswith("Error")
            )
        except Exception as e:
            ros.stop()
            log_tool_call(
                tool_name=self.name,
                parameters=params,
                result=None,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return f"Error moving to pose: {e}"

        log_tool_call(
            tool_name=self.name,
            parameters=params,
            result=result,
            success=success,
            execution_time_ms=(time.time() - start_time) * 1000,
        )
        return result


class ZeroOdometryTool(BaseTool):
    """
    Re-zero odometry by calling the robot-side odom-reset relay service.

    Calls the ``/reset_odom`` (``std_srvs/srv/Trigger``) service exposed by the
    ``odom_reset_relay`` node, which snaps the /odom topic to (0, 0, 0) at the
    robot's current pose. Because the reset happens at the source, every /odom
    subscriber — absolute moves (MoveToPose) and the GetOdometry readout — sees
    the zeroed frame. It does NOT reset any map/AMCL/FASTLIO localization.
    """

    name: str = "ZeroOdometry"
    description: str = (
        "Reset (zero) odometry so the robot's CURRENT pose becomes (0, 0, 0). "
        "Calls the robot-side odom-reset relay service (/reset_odom, "
        "std_srvs/Trigger), which snaps the /odom topic to zero at the current "
        "pose; absolute MoveToPose coordinates and the GetOdometry readout are "
        "then measured from here. Call this to establish a local reference "
        "before giving absolute MoveToPose goals (e.g. operator says 'set this "
        "as the origin' / 'zero the odometry'). It does NOT reset map "
        "localization. No inputs."
    )
    args_schema: Type[BaseModel] = ZeroOdometryInput
    return_direct: bool = False

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Call the odom-reset relay service to zero /odom at the source."""
        start_time = time.time()

        result = get_ros_interface().zero_odometry()

        success = not result.lower().startswith(("error", "odometry reset failed"))
        log_tool_call(
            tool_name=self.name,
            parameters={},
            result=result,
            success=success,
            execution_time_ms=(time.time() - start_time) * 1000,
        )
        return result


class StopRobotTool(BaseTool):
    """Tool to immediately stop the robot."""

    name: str = "StopRobot"
    description: str = (
        "Immediately stop all robot movement. "
        "Use this in emergencies or when you need to halt the robot. "
        "This will cancel any ongoing movement action goals and stop the robot."
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
                ros.stop()  # Cancels active goals + publishes zero velocity
                result = "Robot stopped. Any active movement goals have been cancelled."
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
            result += f" Reason: {reason}"

        execution_time = (time.time() - start_time) * 1000
        log_tool_call(
            tool_name=self.name,
            parameters={"reason": reason},
            result=result,
            success=True,
            execution_time_ms=execution_time,
        )

        return result


class NavigateToPoseTool(BaseTool):
    """
    Autonomously drive to an absolute (x, y, optional yaw) pose using Nav2.

    Sends a goal to the Nav2 ``NavigateToPose`` action server — the same
    obstacle-aware global + local planning as RViz's "2D Nav Goal". Nav2 plans a
    path around obstacles and drives there in closed loop using its own costmaps
    and velocity limits; this is NOT dead-reckoning. The goal defaults to the
    ``map`` frame.

    Relative, dead-reckoning moves (MoveForward/MoveBackward/TurnAngle) are
    handled separately by the custom drive_distance/rotate_angle action server.
    """

    name: str = "NavigateToPose"
    description: str = (
        "PRIMARY NAVIGATION TOOL. Autonomously drive to an absolute pose "
        "(x, y, optional yaw_deg) using Nav2 — the same obstacle-aware global + "
        "local planning as RViz's '2D Nav Goal'. Nav2 plans a path AROUND "
        "obstacles and drives there in closed loop; it does NOT blindly "
        "dead-reckon. USE THIS BY DEFAULT for any navigation request unless the "
        "operator explicitly asks for a relative move in robot frame (e.g. "
        "'forward 2 m'). Goal frame defaults to 'map'. Inputs: x (m), y (m), "
        "yaw_deg (optional; math convention: 0=+X, 90=+Y, CCW positive), frame "
        "(optional; default 'map'). Examples: 'go to (3,0.5)' -> x=3,y=0.5; "
        "'return to origin' -> x=0,y=0,yaw_deg=0."
    )
    args_schema: Type[BaseModel] = NavigateToPoseInput
    return_direct: bool = False

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        x: float,
        y: float,
        yaw_deg: Optional[float] = None,
        frame: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        start_time = time.time()
        ros = get_ros_interface()
        frame_id = frame or NAV2_GOAL_FRAME
        params = {"x": x, "y": y, "yaw_deg": yaw_deg, "frame": frame_id}

        if ros.simulation_mode:
            logger.info(
                f"[SIM] NavigateToPose (Nav2) -> ({x:.2f}, {y:.2f}, "
                f"yaw={yaw_deg}) in '{frame_id}'"
            )
            log_tool_call(
                tool_name=self.name,
                parameters=params,
                result="simulated",
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return (
                f"Navigating to ({x:.2f}, {y:.2f}, yaw={yaw_deg}) via Nav2 in "
                f"'{frame_id}' frame (simulated)"
            )

        yaw_rad = math.radians(yaw_deg) if yaw_deg is not None else None
        try:
            result = ros.navigate_to_pose(
                x=x, y=y, yaw_rad=yaw_rad, frame_id=frame_id
            )
            success = not result.startswith("Error")
        except Exception as e:
            ros.stop()
            log_tool_call(
                tool_name=self.name,
                parameters=params,
                result=None,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return f"Error navigating to pose: {e}"

        log_tool_call(
            tool_name=self.name,
            parameters=params,
            result=result,
            success=success,
            execution_time_ms=(time.time() - start_time) * 1000,
        )
        return result


# Convenience function to create all movement tools
def get_movement_tools() -> list[BaseTool]:
    """Get all movement-related tools."""
    return [
        MoveForwardTool(),
        MoveBackwardTool(),
        TurnAngleTool(),
        MoveToPoseTool(),
        ZeroOdometryTool(),
        StopRobotTool(),
        NavigateToPoseTool(),
    ]
