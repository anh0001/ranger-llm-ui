"""
Tests for Ranger LLM UI tools.

These tests verify the functionality of the robot control tools
in simulation mode (without requiring ROS 2 or actual hardware).
"""

import math
import pytest
import time
from unittest.mock import Mock, patch
import numpy as np

from ranger_llm_ui.tools.movement_tools import (
    MoveForwardTool,
    MoveBackwardTool,
    TurnAngleTool,
    MoveToPoseTool,
    ZeroOdometryTool,
    StopRobotTool,
    get_movement_tools,
    get_ros_interface,
    initialize_ros_interface,
)
from ranger_llm_ui.tools.status_tools import (
    BatteryStatusTool,
    SystemHealthTool,
    GetOdometryTool,
    ListNodesTool,
    ListTopicsTool,
    get_status_tools,
    get_status_interface,
    initialize_status_interface,
)
from ranger_llm_ui.tools.camera_tools import (
    GetCameraImageTool,
    get_camera_tools,
    initialize_camera_interface,
)
from ranger_llm_ui.tools.all_tools import (
    get_all_tools,
    get_tools_by_category,
    initialize_all_tools,
    TOOL_CATEGORIES,
)
from ranger_llm_ui.safety.guard import (
    SafetyGuard,
    SafetyConfig,
    get_safety_guard,
    validate_velocity,
    validate_distance,
)
from ranger_llm_ui.schemas.commands import (
    MoveCommand,
    TurnCommand,
    StopCommand,
    Direction,
    CommandStatus,
)


class TestMovementTools:
    """Test suite for movement tools."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Initialize in simulation mode (no ROS node)
        initialize_ros_interface(None)
        # Reset shared safety state between tests
        guard = get_safety_guard()
        guard.deactivate_emergency_stop()
        guard.state.battery_level = None

    def test_move_forward_basic(self):
        """Test basic forward movement."""
        tool = MoveForwardTool()
        result = tool.run({"distance_m": 1.0})

        assert "forward" in result.lower()
        assert "1.00" in result or "1.0" in result
        assert "simulated" in result.lower()

    def test_move_forward_validates_distance(self):
        """Test that move forward validates distance limits."""
        tool = MoveForwardTool()

        # Should work for reasonable distance
        result = tool.run({"distance_m": 0.5})
        assert "forward" in result.lower()

    def test_move_forward_blocks_when_emergency_stop_active(self):
        """Move commands should be blocked when emergency stop is active."""
        guard = get_safety_guard()
        guard.activate_emergency_stop("test")

        tool = MoveForwardTool()
        result = tool.run({"distance_m": 1.0})

        assert "safety blocked" in result.lower()
        assert "emergency stop" in result.lower()

    def test_move_backward_basic(self):
        """Test basic backward movement."""
        tool = MoveBackwardTool()
        result = tool.run({"distance_m": 0.5})

        assert "backward" in result.lower()
        assert "0.50" in result or "0.5" in result

    def test_turn_angle_right(self):
        """Test turning right (positive angle)."""
        tool = TurnAngleTool()
        result = tool.run({"angle_deg": 90})

        assert "right" in result.lower() or "90" in result
        assert "simulated" in result.lower()

    def test_turn_angle_left(self):
        """Test turning left (negative angle)."""
        tool = TurnAngleTool()
        result = tool.run({"angle_deg": -90})

        assert "left" in result.lower() or "90" in result

    def test_turn_angle_blocks_on_critical_battery(self):
        """Turn commands should be blocked when battery is critically low."""
        guard = get_safety_guard()
        guard.update_battery_level(5.0)

        tool = TurnAngleTool()
        result = tool.run({"angle_deg": 45})

        assert "safety blocked" in result.lower()
        assert "battery" in result.lower()

    def test_stop_robot(self):
        """Test emergency stop."""
        tool = StopRobotTool()
        result = tool.run({})

        assert "stop" in result.lower()

    def test_stop_robot_with_reason(self):
        """Test stop with reason."""
        tool = StopRobotTool()
        result = tool.run({"reason": "User requested stop"})

        assert "stop" in result.lower()
        assert "User requested stop" in result

    def test_get_movement_tools(self):
        """Test getting all movement tools."""
        tools = get_movement_tools()

        assert len(tools) == 7
        tool_names = [t.name for t in tools]
        assert "MoveForward" in tool_names
        assert "MoveBackward" in tool_names
        assert "TurnAngle" in tool_names
        assert "MoveToPose" in tool_names
        assert "ZeroOdometry" in tool_names
        assert "StopRobot" in tool_names
        assert "NavigateToPose" in tool_names


class TestMoveToPoseTool:
    """Test suite for the absolute turn-drive-turn MoveToPose primitive.

    In simulation mode the current pose is always (0, 0, 0), so absolute goals
    here are measured from the origin; absolute planning from a non-origin pose
    is covered by ``test_absolute_plan_from_nonzero_pose``.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures (simulation mode)."""
        initialize_ros_interface(None)
        guard = get_safety_guard()
        guard.deactivate_emergency_stop()
        guard.state.battery_level = None

    def test_forward_only_drives_without_turning(self):
        """A straight-ahead offset with no heading should be a pure drive."""
        tool = MoveToPoseTool()
        result = tool.run({"x": 1.0, "y": 0.0})

        assert "drove 1.00 m" in result
        assert "faced the destination" not in result
        assert "final heading" not in result
        assert "simulated" in result.lower()

    def test_turn_drive_turn_sequence_and_directions(self):
        """An off-axis goal with a heading runs face -> drive -> final turn."""
        tool = MoveToPoseTool()
        # (1, 1) is 45 deg to the left; final heading 90 deg leaves 45 deg left.
        result = tool.run({"x": 1.0, "y": 1.0, "yaw_deg": 90.0})

        assert "faced the destination (turned 45.0° left)" in result
        assert "drove 1.41 m" in result
        assert "turned to the final heading (turned 45.0° left)" in result
        # Phases must be ordered face -> drive -> final heading.
        assert result.index("faced") < result.index("drove") < result.index("final heading")

    def test_right_side_target_turns_right(self):
        """A target on the right (negative y) should turn right (CW)."""
        tool = MoveToPoseTool()
        result = tool.run({"x": 1.0, "y": -1.0})

        assert "faced the destination (turned 45.0° right)" in result
        assert "drove 1.41 m" in result

    def test_pure_rotation_when_no_offset(self):
        """A zero offset with a heading collapses to a single rotation."""
        tool = MoveToPoseTool()
        result = tool.run({"x": 0.0, "y": 0.0, "yaw_deg": 90.0})

        assert "drove" not in result
        assert "turned to the final heading (turned 90.0° left)" in result

    def test_no_op_when_nothing_to_do(self):
        """A zero offset and no heading is reported as a no-op."""
        tool = MoveToPoseTool()
        result = tool.run({"x": 0.0, "y": 0.0})

        assert "no movement needed" in result.lower()

    def test_blocks_when_emergency_stop_active(self):
        """MoveToPose must honor the emergency-stop safety gate."""
        guard = get_safety_guard()
        guard.activate_emergency_stop("test")

        tool = MoveToPoseTool()
        result = tool.run({"x": 1.0, "y": 0.0})

        assert "safety blocked" in result.lower()
        assert "emergency stop" in result.lower()

    def test_large_move_requires_confirmation(self):
        """A travel distance beyond the confirmation threshold is gated."""
        tool = MoveToPoseTool()
        # hypot(3, 3) ~= 4.24 m, beyond the 3 m confirmation threshold.
        result = tool.run({"x": 3.0, "y": 3.0})

        assert "confirmation required" in result.lower()

    def test_absolute_plan_from_nonzero_pose(self):
        """Planning uses the supplied current pose to face/turn in absolute terms."""
        ros = get_ros_interface()
        # Robot at (1, 1) facing +y (90 deg); absolute goal (3, 2) facing +x (0).
        result = ros.move_to_pose(
            3.0, 2.0, yaw_rad=0.0, current_pose=(1.0, 1.0, math.radians(90.0))
        )

        # Bearing to goal = atan2(1, 2) = 26.6 deg; from heading 90 that is a
        # 63.4 deg right turn. Travel = hypot(2, 1) = 2.24 m. Final turn from
        # 26.6 deg to 0 = 26.6 deg right.
        assert "faced the destination (turned 63.4° right)" in result
        assert "drove 2.24 m" in result
        assert "turned to the final heading (turned 26.6° right)" in result


class _ImmediateFuture:
    """A minimal rclpy-future stand-in that is already done with `result`."""

    def __init__(self, result):
        self._result = result

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self._result


class TestZeroOdometry:
    """Test suite for the service-backed ZeroOdometry tool."""

    @pytest.fixture(autouse=True)
    def setup(self):
        initialize_ros_interface(None)

    def test_zero_odometry_tool_simulated(self):
        """ZeroOdometry succeeds (and is reported as simulated) without ROS."""
        result = ZeroOdometryTool().run({})
        assert "reset" in result.lower()
        assert "simulated" in result.lower()

    def test_zero_odometry_calls_trigger_service(self):
        """In real mode it calls /reset_odom and returns the service message."""
        ros = get_ros_interface()
        orig_mode = ros._simulation_mode
        orig_client = ros._reset_odom_client
        try:
            ros._simulation_mode = False
            response = Mock()
            response.success = True
            response.message = "Odometry zeroed at raw pose (x=1.0, y=2.0, yaw=0.5)."
            client = Mock()
            client.wait_for_service.return_value = True
            client.call_async.return_value = _ImmediateFuture(response)
            ros._reset_odom_client = client

            result = ros.zero_odometry()

            client.call_async.assert_called_once()
            assert "odometry zeroed at raw pose" in result.lower()
            assert not result.lower().startswith("error")
        finally:
            ros._simulation_mode = orig_mode
            ros._reset_odom_client = orig_client

    def test_zero_odometry_reports_service_unavailable(self):
        """A missing relay service is reported as an actionable error."""
        ros = get_ros_interface()
        orig_mode = ros._simulation_mode
        orig_client = ros._reset_odom_client
        try:
            ros._simulation_mode = False
            client = Mock()
            client.wait_for_service.return_value = False
            ros._reset_odom_client = client

            result = ros.zero_odometry()

            assert result.lower().startswith("error")
            assert "reset_odom" in result
            client.call_async.assert_not_called()
        finally:
            ros._simulation_mode = orig_mode
            ros._reset_odom_client = orig_client

    def test_zero_odometry_failure_response(self):
        """A success=False Trigger response is surfaced as a failure message."""
        ros = get_ros_interface()
        orig_mode = ros._simulation_mode
        orig_client = ros._reset_odom_client
        try:
            ros._simulation_mode = False
            response = Mock()
            response.success = False
            response.message = "No odometry received yet on /odom_raw."
            client = Mock()
            client.wait_for_service.return_value = True
            client.call_async.return_value = _ImmediateFuture(response)
            ros._reset_odom_client = client

            result = ros.zero_odometry()

            assert "failed" in result.lower()
            assert "no odometry received yet" in result.lower()
        finally:
            ros._simulation_mode = orig_mode
            ros._reset_odom_client = orig_client


class TestStatusTools:
    """Test suite for status tools."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        initialize_status_interface(None)

    def test_battery_status(self):
        """Test battery status query."""
        tool = BatteryStatusTool()
        result = tool.run({})

        assert "battery" in result.lower()
        assert "%" in result

    def test_system_health(self):
        """Test system health check."""
        tool = SystemHealthTool()
        result = tool.run({})

        assert "health" in result.lower() or "status" in result.lower()
        assert "simulation" in result.lower()

    def test_system_health_reports_battery_unavailable(self):
        """System health should explicitly report missing battery data."""
        interface = get_status_interface()
        original_mode = interface._simulation_mode
        original_battery = interface._battery_state

        try:
            interface._simulation_mode = False
            interface._battery_state = None

            tool = SystemHealthTool()
            result = tool.run({})
            assert "battery status unavailable" in result.lower()
        finally:
            interface._simulation_mode = original_mode
            interface._battery_state = original_battery

    def test_get_odometry(self):
        """Test odometry query."""
        tool = GetOdometryTool()
        result = tool.run({})

        assert "position" in result.lower()
        assert "x" in result.lower()
        assert "y" in result.lower()

    def test_list_nodes(self):
        """Test listing ROS nodes."""
        tool = ListNodesTool()
        result = tool.run({})

        assert "node" in result.lower()

    def test_list_topics(self):
        """Test listing ROS topics."""
        tool = ListTopicsTool()
        result = tool.run({})

        assert "topic" in result.lower()

    def test_get_status_tools(self):
        """Test getting all status tools."""
        tools = get_status_tools()

        assert len(tools) == 5
        tool_names = [t.name for t in tools]
        assert "BatteryStatus" in tool_names
        assert "SystemHealth" in tool_names

    def test_battery_level_normalizes_fractional_percentage(self):
        """Battery percentages in 0..1 range should be normalized to 0..100."""
        interface = get_status_interface()
        original_mode = interface._simulation_mode
        original_battery = interface._battery_state

        try:
            interface._simulation_mode = False
            msg = Mock()
            msg.voltage = 24.5
            msg.percentage = 0.85
            msg.power_supply_status = 2
            interface._battery_state = msg

            level, status, voltage_v = interface.get_battery_level()
            assert level == pytest.approx(85.0)
            assert status == "discharging"
            assert voltage_v == pytest.approx(24.5)
        finally:
            interface._simulation_mode = original_mode
            interface._battery_state = original_battery


class TestCameraTools:
    """Test suite for camera tools."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        initialize_camera_interface(None)

    def test_get_camera_image(self):
        """Test camera image retrieval (simulation)."""
        tool = GetCameraImageTool()
        result = tool.run({"format": "png"})

        assert "camera image" in result.lower()
        assert "data:image/png;base64" in result

    def test_get_camera_tools(self):
        """Test getting all camera tools."""
        tools = get_camera_tools()

        assert len(tools) == 1
        assert tools[0].name == "GetCameraImage"

    def test_yuyv422_conversion_to_rgb(self):
        """Test direct YUYV422 decoding to RGB."""
        interface = initialize_camera_interface(None)
        msg = Mock()
        msg.width = 2
        msg.height = 2
        msg.step = 4
        # YUYV bytes: [Y0, U, Y1, V] with neutral chroma (U=V=128).
        msg.data = bytes([50, 128, 100, 128, 150, 128, 200, 128])

        image = interface._convert_yuv422_to_rgb(msg, "yuyv")

        assert image is not None
        assert image.shape == (2, 2, 3)
        assert image.dtype == np.uint8
        assert np.all(np.abs(image[:, :, 0] - np.array([[50, 100], [150, 200]])) <= 1)
        assert np.all(np.abs(image[:, :, 1] - np.array([[50, 100], [150, 200]])) <= 1)
        assert np.all(np.abs(image[:, :, 2] - np.array([[50, 100], [150, 200]])) <= 1)

    def test_yuv422_fallback_when_primary_converter_fails(self):
        """Test conversion fallback for yuv422 encoding."""
        interface = initialize_camera_interface(None)
        msg = Mock()
        msg.width = 2
        msg.height = 2
        msg.step = 4
        msg.encoding = "yuv422"
        # UYVY bytes: [U, Y0, V, Y1] with neutral chroma (U=V=128).
        msg.data = bytes([128, 60, 128, 120, 128, 180, 128, 240])

        image = interface._convert_ros_image(msg)

        assert image is not None
        assert image.shape == (2, 2, 3)
        assert np.all(np.abs(image[:, :, 0] - np.array([[60, 120], [180, 240]])) <= 1)


class TestAllTools:
    """Test suite for tool registry."""

    def test_get_all_tools(self):
        """Test getting all tools."""
        tools = get_all_tools()

        assert len(tools) > 0
        # Should have movement, status, and diagnostic tools
        tool_names = [t.name for t in tools]
        assert "MoveForward" in tool_names
        assert "BatteryStatus" in tool_names
        assert "ListNodes" in tool_names
        assert "GetCameraImage" in tool_names

    def test_get_tools_by_category(self):
        """Test getting tools by category."""
        movement_tools = get_tools_by_category("movement")
        assert len(movement_tools) == 7

        status_tools = get_tools_by_category("status")
        assert len(status_tools) == 3

        perception_tools = get_tools_by_category("perception")
        assert len(perception_tools) == 2

        with pytest.raises(ValueError):
            get_tools_by_category("invalid_category")

    def test_tool_categories(self):
        """Test that tool categories are defined correctly."""
        assert "movement" in TOOL_CATEGORIES
        assert "status" in TOOL_CATEGORIES
        assert "diagnostics" in TOOL_CATEGORIES
        assert "perception" in TOOL_CATEGORIES


class TestSafetyGuard:
    """Test suite for safety guard."""

    def test_velocity_clamping(self):
        """Test velocity is clamped to safe limits."""
        guard = SafetyGuard()

        # Test linear velocity
        assert guard.validate_and_clamp_velocity(0.1) == 0.1
        assert guard.validate_and_clamp_velocity(1.0) == 0.5  # Clamped to max
        assert guard.validate_and_clamp_velocity(-0.3) == -0.3

        # Test angular velocity
        assert guard.validate_and_clamp_velocity(0.5, is_angular=True) == 0.5
        assert guard.validate_and_clamp_velocity(2.0, is_angular=True) == 1.0  # Clamped

    def test_distance_validation(self):
        """Test distance validation."""
        guard = SafetyGuard()

        # Normal distance
        dist, needs_confirm, msg = guard.validate_distance(1.0)
        assert dist == 1.0
        assert not needs_confirm

        # Large distance requiring confirmation
        dist, needs_confirm, msg = guard.validate_distance(4.0)
        assert dist == 4.0
        assert needs_confirm
        assert "confirmation" in msg.lower()

        # Exceeds max - should be clamped
        dist, needs_confirm, msg = guard.validate_distance(10.0)
        assert dist == 5.0  # Clamped to max
        assert needs_confirm

    def test_emergency_stop(self):
        """Test emergency stop functionality."""
        guard = SafetyGuard()

        assert not guard.state.emergency_stop_active

        guard.activate_emergency_stop("Test")
        assert guard.state.emergency_stop_active

        is_safe, reason = guard.is_safe_to_operate()
        assert not is_safe
        assert "emergency" in reason.lower()

        guard.deactivate_emergency_stop()
        assert not guard.state.emergency_stop_active

    def test_battery_level_tracking(self):
        """Test battery level tracking."""
        guard = SafetyGuard()

        guard.update_battery_level(50.0)
        assert guard.state.battery_level == 50.0

        # Normal battery - should be safe
        is_safe, _ = guard.is_safe_to_operate()
        assert is_safe

        # Critical battery - should not be safe
        guard.update_battery_level(5.0)
        is_safe, reason = guard.is_safe_to_operate()
        assert not is_safe
        assert "battery" in reason.lower()

    def test_command_safety_check(self):
        """Test comprehensive command safety check."""
        guard = SafetyGuard()

        # Normal move command
        is_safe, msg, params = guard.check_command_safety("move", distance_m=1.0)
        assert is_safe
        assert "distance_m" in params

        # Stop command is always safe
        is_safe, msg, params = guard.check_command_safety("stop")
        assert is_safe


class TestCommandSchemas:
    """Test suite for command schemas."""

    def test_move_command(self):
        """Test MoveCommand dataclass."""
        cmd = MoveCommand(
            direction=Direction.FORWARD,
            distance_m=1.5,
            velocity_mps=0.2,
        )

        assert cmd.direction == Direction.FORWARD
        assert cmd.distance_m == 1.5
        assert cmd.status == CommandStatus.PENDING

        # Test to_dict
        d = cmd.to_dict()
        assert d["direction"] == "forward"
        assert d["distance_m"] == 1.5

    def test_turn_command(self):
        """Test TurnCommand dataclass."""
        cmd = TurnCommand(angle_deg=90.0)

        assert cmd.angle_deg == 90.0
        assert cmd.status == CommandStatus.PENDING

    def test_stop_command(self):
        """Test StopCommand dataclass."""
        cmd = StopCommand(reason="User request", emergency=False)

        assert cmd.reason == "User request"
        assert not cmd.emergency

    def test_command_status_transitions(self):
        """Test command status can be updated."""
        cmd = MoveCommand(distance_m=1.0)

        assert cmd.status == CommandStatus.PENDING

        cmd.status = CommandStatus.EXECUTING
        assert cmd.status == CommandStatus.EXECUTING

        cmd.status = CommandStatus.COMPLETED
        assert cmd.status == CommandStatus.COMPLETED


class TestIntegration:
    """Integration tests for the tool system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up integration test fixtures."""
        initialize_all_tools(None)

    def test_tool_chain_movement(self):
        """Test a chain of movement commands."""
        forward = MoveForwardTool()
        turn = TurnAngleTool()
        stop = StopRobotTool()

        # Execute movement sequence
        result1 = forward.run({"distance_m": 1.0})
        assert "forward" in result1.lower()

        result2 = turn.run({"angle_deg": 90})
        assert "90" in result2 or "right" in result2.lower()

        result3 = stop.run({})
        assert "stop" in result3.lower()

    def test_status_after_movement(self):
        """Test getting status after movement."""
        forward = MoveForwardTool()
        battery = BatteryStatusTool()
        odom = GetOdometryTool()

        # Move
        forward.run({"distance_m": 0.5})

        # Check status
        battery_result = battery.run({})
        assert "battery" in battery_result.lower()

        odom_result = odom.run({})
        assert "position" in odom_result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
