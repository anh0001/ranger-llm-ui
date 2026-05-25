"""
Movement Action Server - ROS 2 action servers for robot movement.

Hosts two action servers:
  - /drive_distance (DriveDistance) — closed-loop linear movement
  - /rotate_angle (RotateAngle) — closed-loop rotation in place

Uses odometry feedback for accurate, cancellable movement control.

Usage:
    ros2 run ranger_llm_ui movement_server
"""

import math
import time
import logging
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from ranger_llm_msgs.action import DriveDistance, RotateAngle

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONTROL_RATE_HZ = 20.0
DEFAULT_DISTANCE_TOLERANCE_M = 0.05
DEFAULT_ANGLE_TOLERANCE_DEG = 1.0
DEFAULT_MAX_LINEAR_VELOCITY = 0.5   # m/s
DEFAULT_MAX_ANGULAR_VELOCITY = 1.0  # rad/s (≈57 deg/s)
DEFAULT_MIN_VELOCITY = 0.05         # m/s minimum to overcome friction
DEFAULT_MIN_ANGULAR_VELOCITY = 0.1  # rad/s minimum

# Proportional controller gain
LINEAR_KP = 1.0
ANGULAR_KP = 2.0

# Velocity ramp-down distance/angle thresholds
LINEAR_RAMP_DISTANCE_M = 0.3
ANGULAR_RAMP_ANGLE_DEG = 15.0


def _quaternion_to_yaw(orientation):
    """Extract yaw angle from quaternion orientation."""
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle(angle_rad):
    """Normalize angle to [-pi, pi]."""
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


class MovementActionServer(Node):
    """ROS 2 node hosting drive_distance and rotate_angle action servers."""

    def __init__(self, node_name='ranger_movement_server'):
        super().__init__(node_name)

        self._cb_group = ReentrantCallbackGroup()

        # Publisher for velocity commands
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Odometry subscriber
        self._current_odom = None
        self._odom_lock = threading.Lock()
        self._odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10,
            callback_group=self._cb_group,
        )

        # Mutex: only one movement goal at a time
        self._goal_lock = threading.Lock()

        # Action servers
        self._drive_server = ActionServer(
            self,
            DriveDistance,
            'drive_distance',
            execute_callback=self._execute_drive,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )

        self._rotate_server = ActionServer(
            self,
            RotateAngle,
            'rotate_angle',
            execute_callback=self._execute_rotate,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )

        self.get_logger().info('Movement action server ready')

    # ── Callbacks ──────────────────────────────────────────────

    def _odom_callback(self, msg):
        with self._odom_lock:
            self._current_odom = msg

    def _goal_callback(self, goal_request):
        """Accept all goals (single-goal enforcement is in execute)."""
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        """Always accept cancel requests."""
        self.get_logger().info('Cancel requested — stopping robot')
        self._stop_robot()
        return CancelResponse.ACCEPT

    # ── Helpers ────────────────────────────────────────────────

    def _get_odom(self):
        with self._odom_lock:
            return self._current_odom

    def _get_position(self, odom):
        """Extract (x, y) from Odometry message."""
        p = odom.pose.pose.position
        return p.x, p.y

    def _get_yaw(self, odom):
        """Extract yaw from Odometry message."""
        return _quaternion_to_yaw(odom.pose.pose.orientation)

    def _publish_velocity(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(msg)

    def _stop_robot(self):
        """Publish zero velocity multiple times to ensure delivery."""
        for _ in range(3):
            self._publish_velocity(0.0, 0.0)
            time.sleep(0.05)

    # ── Drive Distance Execute ─────────────────────────────────

    def _execute_drive(self, goal_handle):
        """Execute a DriveDistance goal with closed-loop odometry control."""
        if not self._goal_lock.acquire(blocking=False):
            goal_handle.abort()
            result = DriveDistance.Result()
            result.success = False
            result.message = 'Another movement goal is already active'
            return result

        try:
            return self._do_drive(goal_handle)
        finally:
            self._goal_lock.release()

    def _do_drive(self, goal_handle):
        goal = goal_handle.request
        target_distance = abs(goal.distance_m)
        direction = 1.0 if goal.distance_m >= 0 else -1.0
        max_vel = min(abs(goal.max_velocity_mps), DEFAULT_MAX_LINEAR_VELOCITY)
        max_vel = max(max_vel, DEFAULT_MIN_VELOCITY)

        result = DriveDistance.Result()
        feedback = DriveDistance.Feedback()

        # Wait for initial odometry
        start_odom = self._get_odom()
        if start_odom is None:
            self.get_logger().warn('Waiting for odometry...')
            for _ in range(50):  # 2.5s at 20Hz
                time.sleep(0.05)
                start_odom = self._get_odom()
                if start_odom is not None:
                    break
            if start_odom is None:
                goal_handle.abort()
                result.success = False
                result.message = 'No odometry data available'
                return result

        start_x, start_y = self._get_position(start_odom)
        start_time = time.monotonic()
        rate_period = 1.0 / DEFAULT_CONTROL_RATE_HZ

        self.get_logger().info(
            f'Driving {goal.distance_m:.2f}m at max {max_vel:.2f} m/s'
        )

        while True:
            # Check cancellation
            if goal_handle.is_cancel_requested:
                self._stop_robot()
                goal_handle.canceled()
                elapsed = time.monotonic() - start_time
                odom = self._get_odom() or start_odom
                cx, cy = self._get_position(odom)
                traveled = math.sqrt((cx - start_x)**2 + (cy - start_y)**2)
                result.distance_traveled_m = traveled
                result.elapsed_time_s = elapsed
                result.success = False
                result.message = f'Cancelled after {traveled:.2f}m'
                return result

            # Compute distance traveled
            odom = self._get_odom() or start_odom
            cx, cy = self._get_position(odom)
            traveled = math.sqrt((cx - start_x)**2 + (cy - start_y)**2)
            remaining = target_distance - traveled
            elapsed = time.monotonic() - start_time

            # Goal reached?
            if remaining <= DEFAULT_DISTANCE_TOLERANCE_M:
                self._stop_robot()
                goal_handle.succeed()
                result.distance_traveled_m = traveled
                result.elapsed_time_s = elapsed
                result.success = True
                result.message = f'Drove {traveled:.2f}m in {elapsed:.1f}s'
                return result

            # Timeout safety (3x expected time + 5s buffer)
            expected_time = target_distance / max_vel
            if elapsed > expected_time * 3.0 + 5.0:
                self._stop_robot()
                goal_handle.abort()
                result.distance_traveled_m = traveled
                result.elapsed_time_s = elapsed
                result.success = False
                result.message = f'Timeout after {elapsed:.1f}s (traveled {traveled:.2f}m)'
                return result

            # P-controller with ramp-down
            if remaining < LINEAR_RAMP_DISTANCE_M:
                velocity = max(
                    DEFAULT_MIN_VELOCITY,
                    max_vel * (remaining / LINEAR_RAMP_DISTANCE_M) * LINEAR_KP,
                )
            else:
                velocity = max_vel

            self._publish_velocity(linear_x=direction * velocity)

            # Publish feedback
            feedback.distance_remaining_m = remaining
            feedback.elapsed_time_s = elapsed
            feedback.current_velocity_mps = velocity
            goal_handle.publish_feedback(feedback)

            time.sleep(rate_period)

    # ── Rotate Angle Execute ───────────────────────────────────

    def _execute_rotate(self, goal_handle):
        """Execute a RotateAngle goal with closed-loop odometry control."""
        if not self._goal_lock.acquire(blocking=False):
            goal_handle.abort()
            result = RotateAngle.Result()
            result.success = False
            result.message = 'Another movement goal is already active'
            return result

        try:
            return self._do_rotate(goal_handle)
        finally:
            self._goal_lock.release()

    def _do_rotate(self, goal_handle):
        goal = goal_handle.request
        target_angle_deg = abs(goal.angle_deg)
        target_angle_rad = math.radians(target_angle_deg)
        tolerance_rad = math.radians(DEFAULT_ANGLE_TOLERANCE_DEG)

        # Positive angle_deg = clockwise = negative angular.z in ROS
        direction = -1.0 if goal.angle_deg > 0 else 1.0

        max_ang_vel_dps = min(
            abs(goal.max_angular_velocity_dps),
            math.degrees(DEFAULT_MAX_ANGULAR_VELOCITY),
        )
        max_ang_vel_dps = max(max_ang_vel_dps, math.degrees(DEFAULT_MIN_ANGULAR_VELOCITY))
        max_ang_vel_rad = math.radians(max_ang_vel_dps)

        result = RotateAngle.Result()
        feedback = RotateAngle.Feedback()

        # Wait for initial odometry
        start_odom = self._get_odom()
        if start_odom is None:
            self.get_logger().warn('Waiting for odometry...')
            for _ in range(50):
                time.sleep(0.05)
                start_odom = self._get_odom()
                if start_odom is not None:
                    break
            if start_odom is None:
                goal_handle.abort()
                result.success = False
                result.message = 'No odometry data available'
                return result

        start_yaw = self._get_yaw(start_odom)
        start_time = time.monotonic()
        rate_period = 1.0 / DEFAULT_CONTROL_RATE_HZ

        self.get_logger().info(
            f'Rotating {goal.angle_deg:.1f}° at max {max_ang_vel_dps:.1f} deg/s'
        )

        while True:
            # Check cancellation
            if goal_handle.is_cancel_requested:
                self._stop_robot()
                goal_handle.canceled()
                elapsed = time.monotonic() - start_time
                odom = self._get_odom() or start_odom
                current_yaw = self._get_yaw(odom)
                rotated_rad = abs(_normalize_angle(current_yaw - start_yaw))
                result.angle_rotated_deg = math.degrees(rotated_rad)
                result.elapsed_time_s = elapsed
                result.success = False
                result.message = f'Cancelled after {math.degrees(rotated_rad):.1f}°'
                return result

            # Compute angle rotated
            odom = self._get_odom() or start_odom
            current_yaw = self._get_yaw(odom)
            rotated_rad = abs(_normalize_angle(current_yaw - start_yaw))
            remaining_rad = target_angle_rad - rotated_rad
            remaining_deg = math.degrees(remaining_rad)
            elapsed = time.monotonic() - start_time

            # Goal reached?
            if remaining_rad <= tolerance_rad:
                self._stop_robot()
                goal_handle.succeed()
                result.angle_rotated_deg = math.degrees(rotated_rad)
                result.elapsed_time_s = elapsed
                result.success = True
                result.message = (
                    f'Rotated {math.degrees(rotated_rad):.1f}° in {elapsed:.1f}s'
                )
                return result

            # Timeout safety
            expected_time = target_angle_rad / max_ang_vel_rad
            if elapsed > expected_time * 3.0 + 5.0:
                self._stop_robot()
                goal_handle.abort()
                result.angle_rotated_deg = math.degrees(rotated_rad)
                result.elapsed_time_s = elapsed
                result.success = False
                result.message = (
                    f'Timeout after {elapsed:.1f}s '
                    f'(rotated {math.degrees(rotated_rad):.1f}°)'
                )
                return result

            # P-controller with ramp-down
            ramp_rad = math.radians(ANGULAR_RAMP_ANGLE_DEG)
            if remaining_rad < ramp_rad:
                ang_vel = max(
                    DEFAULT_MIN_ANGULAR_VELOCITY,
                    max_ang_vel_rad * (remaining_rad / ramp_rad) * ANGULAR_KP,
                )
            else:
                ang_vel = max_ang_vel_rad

            self._publish_velocity(angular_z=direction * ang_vel)

            # Publish feedback
            feedback.angle_remaining_deg = remaining_deg
            feedback.elapsed_time_s = elapsed
            feedback.current_angular_velocity_dps = math.degrees(ang_vel)
            goal_handle.publish_feedback(feedback)

            time.sleep(rate_period)


def main(args=None):
    rclpy.init(args=args)
    node = MovementActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
