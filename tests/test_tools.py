"""
Tests for Ranger LLM UI tools.

These tests verify the functionality of the robot control tools
in simulation mode (without requiring ROS 2 or actual hardware).
"""

import pytest
import time
from unittest.mock import Mock, patch

from ranger_llm_ui.tools.movement_tools import (
    MoveForwardTool,
    MoveBackwardTool,
    TurnAngleTool,
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
from ranger_llm_ui.tools.all_tools import (
    get_all_tools,
    get_tools_by_category,
    initialize_all_tools,
    TOOL_CATEGORIES,
)
from ranger_llm_ui.safety.guard import (
    SafetyGuard,
    SafetyConfig,
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

        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        assert "MoveForward" in tool_names
        assert "MoveBackward" in tool_names
        assert "TurnAngle" in tool_names
        assert "StopRobot" in tool_names


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

    def test_get_tools_by_category(self):
        """Test getting tools by category."""
        movement_tools = get_tools_by_category("movement")
        assert len(movement_tools) == 4

        status_tools = get_tools_by_category("status")
        assert len(status_tools) == 3

        with pytest.raises(ValueError):
            get_tools_by_category("invalid_category")

    def test_tool_categories(self):
        """Test that tool categories are defined correctly."""
        assert "movement" in TOOL_CATEGORIES
        assert "status" in TOOL_CATEGORIES
        assert "diagnostics" in TOOL_CATEGORIES


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
