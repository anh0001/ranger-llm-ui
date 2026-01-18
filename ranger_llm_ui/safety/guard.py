"""
Safety Guard - Safety checks and interlocks for robot operations.

This module provides safety constraints at multiple levels:
- Velocity and acceleration limits
- Distance validation
- Confirmation requirements for high-impact actions
- Emergency stop functionality
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Configuration for safety limits and thresholds."""

    # Linear velocity limits (m/s)
    max_linear_velocity: float = 0.5
    min_linear_velocity: float = 0.05
    default_linear_velocity: float = 0.2

    # Angular velocity limits (rad/s)
    max_angular_velocity: float = 1.0
    min_angular_velocity: float = 0.1
    default_angular_velocity: float = 0.5

    # Distance limits (m)
    max_single_move_distance: float = 5.0
    confirmation_threshold_distance: float = 2.0

    # Angle limits (degrees)
    max_single_turn_angle: float = 360.0

    # Timeout limits (seconds)
    default_command_timeout: float = 30.0
    max_command_timeout: float = 120.0

    # Safety thresholds
    low_battery_threshold: float = 20.0  # Warn below this %
    critical_battery_threshold: float = 10.0  # Stop operations below this %

    # Confirmation requirements
    require_confirmation_for_large_moves: bool = True
    require_confirmation_threshold_m: float = 3.0


@dataclass
class SafetyState:
    """Current safety state of the robot."""
    emergency_stop_active: bool = False
    battery_level: Optional[float] = None
    is_moving: bool = False
    last_error: Optional[str] = None
    confirmation_pending: bool = False
    pending_action: Optional[dict] = None


class SafetyGuard:
    """
    Safety guard that validates and constrains robot commands.

    This class provides safety checks before executing any robot action,
    enforcing velocity limits, distance constraints, and confirmation
    requirements for potentially dangerous operations.
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.state = SafetyState()
        self._confirmation_callback: Optional[Callable[[str], bool]] = None

    def set_confirmation_callback(self, callback: Callable[[str], bool]):
        """Set callback function for requesting user confirmation."""
        self._confirmation_callback = callback

    def activate_emergency_stop(self, reason: str = ""):
        """Activate emergency stop."""
        self.state.emergency_stop_active = True
        self.state.last_error = f"Emergency stop activated: {reason}"
        logger.warning(f"EMERGENCY STOP: {reason}")

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop (requires explicit user action)."""
        self.state.emergency_stop_active = False
        self.state.last_error = None
        logger.info("Emergency stop deactivated")

    def is_safe_to_operate(self) -> tuple[bool, str]:
        """
        Check if it's safe to execute commands.

        Returns:
            Tuple of (is_safe, reason_if_not_safe)
        """
        if self.state.emergency_stop_active:
            return False, "Emergency stop is active"

        if self.state.battery_level is not None:
            if self.state.battery_level < self.config.critical_battery_threshold:
                return False, f"Battery critically low ({self.state.battery_level:.1f}%)"

        return True, ""

    def update_battery_level(self, level: float):
        """Update current battery level."""
        self.state.battery_level = level
        if level < self.config.low_battery_threshold:
            logger.warning(f"Low battery warning: {level:.1f}%")

    def validate_and_clamp_velocity(
        self, velocity: float, is_angular: bool = False
    ) -> float:
        """
        Validate and clamp velocity to safe limits.

        Args:
            velocity: Requested velocity
            is_angular: True for angular velocity (rad/s), False for linear (m/s)

        Returns:
            Clamped velocity within safe limits
        """
        if is_angular:
            max_vel = self.config.max_angular_velocity
            min_vel = self.config.min_angular_velocity
        else:
            max_vel = self.config.max_linear_velocity
            min_vel = self.config.min_linear_velocity

        # Preserve sign, clamp magnitude
        sign = 1 if velocity >= 0 else -1
        magnitude = abs(velocity)

        if magnitude < min_vel:
            clamped = min_vel
        elif magnitude > max_vel:
            clamped = max_vel
            logger.warning(
                f"Velocity {velocity} exceeds max {max_vel}, clamping to {sign * clamped}"
            )
        else:
            clamped = magnitude

        return sign * clamped

    def validate_distance(self, distance_m: float) -> tuple[float, bool, str]:
        """
        Validate movement distance.

        Args:
            distance_m: Requested distance in meters

        Returns:
            Tuple of (validated_distance, requires_confirmation, message)
        """
        distance_m = abs(distance_m)

        if distance_m > self.config.max_single_move_distance:
            return (
                self.config.max_single_move_distance,
                True,
                f"Distance {distance_m}m exceeds max {self.config.max_single_move_distance}m, "
                f"clamped to max. Confirmation required.",
            )

        if (
            self.config.require_confirmation_for_large_moves
            and distance_m > self.config.require_confirmation_threshold_m
        ):
            return (
                distance_m,
                True,
                f"Large movement ({distance_m}m) requires confirmation.",
            )

        return distance_m, False, ""

    def validate_angle(self, angle_deg: float) -> tuple[float, str]:
        """
        Validate turn angle.

        Args:
            angle_deg: Requested angle in degrees

        Returns:
            Tuple of (validated_angle, message)
        """
        # Normalize to -360 to 360
        while angle_deg > 360:
            angle_deg -= 360
        while angle_deg < -360:
            angle_deg += 360

        return angle_deg, ""

    def check_command_safety(
        self, command_type: str, **kwargs
    ) -> tuple[bool, str, dict]:
        """
        Perform comprehensive safety check for a command.

        Args:
            command_type: Type of command ("move", "turn", "stop", etc.)
            **kwargs: Command parameters

        Returns:
            Tuple of (is_safe, message, validated_params)
        """
        # First check if we can operate at all
        can_operate, reason = self.is_safe_to_operate()
        if not can_operate:
            return False, reason, {}

        validated_params = dict(kwargs)

        if command_type == "move":
            distance = kwargs.get("distance_m", 0)
            validated_distance, needs_confirm, msg = self.validate_distance(distance)
            validated_params["distance_m"] = validated_distance

            velocity = kwargs.get("velocity_mps", self.config.default_linear_velocity)
            validated_params["velocity_mps"] = self.validate_and_clamp_velocity(
                velocity, is_angular=False
            )

            if needs_confirm:
                return False, msg, validated_params

        elif command_type == "turn":
            angle = kwargs.get("angle_deg", 0)
            validated_angle, msg = self.validate_angle(angle)
            validated_params["angle_deg"] = validated_angle

            angular_vel = kwargs.get(
                "angular_velocity_dps",
                self.config.default_angular_velocity * 57.2958,  # rad/s to deg/s
            )
            # Convert to rad/s for validation, then back
            validated_rad = self.validate_and_clamp_velocity(
                angular_vel / 57.2958, is_angular=True
            )
            validated_params["angular_velocity_dps"] = validated_rad * 57.2958

        elif command_type == "stop":
            # Stop is always safe
            pass

        return True, "", validated_params

    def request_confirmation(self, action_description: str) -> bool:
        """
        Request user confirmation for a potentially dangerous action.

        Args:
            action_description: Description of the action requiring confirmation

        Returns:
            True if confirmed, False otherwise
        """
        if self._confirmation_callback:
            return self._confirmation_callback(action_description)

        # If no callback set, log warning and deny
        logger.warning(
            f"Confirmation required but no callback set: {action_description}"
        )
        return False


# Module-level convenience functions
_default_guard: Optional[SafetyGuard] = None


def get_safety_guard() -> SafetyGuard:
    """Get or create the default safety guard instance."""
    global _default_guard
    if _default_guard is None:
        _default_guard = SafetyGuard()
    return _default_guard


def validate_velocity(velocity: float, is_angular: bool = False) -> float:
    """Validate and clamp velocity using default guard."""
    return get_safety_guard().validate_and_clamp_velocity(velocity, is_angular)


def validate_distance(distance_m: float) -> tuple[float, bool, str]:
    """Validate distance using default guard."""
    return get_safety_guard().validate_distance(distance_m)


def requires_confirmation(command_type: str, **kwargs) -> bool:
    """Check if a command requires user confirmation."""
    _, msg, _ = get_safety_guard().check_command_safety(command_type, **kwargs)
    return "confirmation" in msg.lower()
