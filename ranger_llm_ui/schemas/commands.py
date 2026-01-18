"""
Command Schemas - Dataclass definitions for robot commands.

These schemas define the structure of commands that can be sent to the robot,
ensuring consistency between the agent, tools, and ROS interfaces.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


class Direction(str, Enum):
    """Direction enum for movement commands."""
    FORWARD = "forward"
    BACKWARD = "backward"


class TurnDirection(str, Enum):
    """Turn direction enum."""
    LEFT = "left"
    RIGHT = "right"
    CLOCKWISE = "clockwise"
    COUNTERCLOCKWISE = "counterclockwise"


class CommandStatus(str, Enum):
    """Status of a command execution."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUIRES_CONFIRMATION = "requires_confirmation"


@dataclass
class BaseCommand:
    """Base class for all robot commands."""
    timestamp: datetime = field(default_factory=datetime.now)
    status: CommandStatus = CommandStatus.PENDING
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert command to dictionary for logging."""
        return {
            "type": self.__class__.__name__,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "error_message": self.error_message,
        }


@dataclass
class MoveCommand(BaseCommand):
    """
    Move the robot linearly.

    Implementation: Calls a ROS 2 Action /drive_distance if available,
    or publishes on /cmd_vel with a timed controller.

    Example: {"action": "move", "direction": "forward", "distance_m": 1.0}
    """
    direction: Direction = Direction.FORWARD
    distance_m: float = 0.0
    velocity_mps: Optional[float] = None  # If None, use default safe velocity

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "direction": self.direction.value,
            "distance_m": self.distance_m,
            "velocity_mps": self.velocity_mps,
        })
        return base


@dataclass
class TurnCommand(BaseCommand):
    """
    Rotate the robot in place.

    Implementation: Calls a /rotate_angle service or publishes angular Twist.

    Example: {"action": "turn", "angle_deg": 90}
    """
    angle_deg: float = 0.0
    angular_velocity_dps: Optional[float] = None  # degrees per second

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "angle_deg": self.angle_deg,
            "angular_velocity_dps": self.angular_velocity_dps,
        })
        return base


@dataclass
class StopCommand(BaseCommand):
    """
    Immediate stop command.

    Implementation: Publishes zero velocities to /cmd_vel and/or calls
    an emergency stop service on the motor controller. Also cancels any
    ongoing Move/Turn action.
    """
    reason: Optional[str] = None
    emergency: bool = False  # True for E-stop, False for normal stop

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "reason": self.reason,
            "emergency": self.emergency,
        })
        return base


@dataclass
class StatusCommand(BaseCommand):
    """
    Request robot status information.

    Can be used to query battery, system health, sensor status, etc.
    """
    query_type: str = "all"  # "battery", "system", "sensors", "all"

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "query_type": self.query_type,
        })
        return base


@dataclass
class NavigationCommand(BaseCommand):
    """
    High-level goal navigation command (future scope).

    Implementation: Interface with Nav2 action server.
    This is marked for future implementation with heavy safety gating.
    """
    destination: str = ""  # Named location or coordinates
    coordinates: Optional[tuple[float, float, float]] = None  # x, y, theta

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "destination": self.destination,
            "coordinates": self.coordinates,
        })
        return base


@dataclass
class CommandResult:
    """Result of a command execution."""
    success: bool
    message: str
    command: BaseCommand
    execution_time_s: float = 0.0
    data: Optional[dict] = None  # Additional data returned by the command

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "command": self.command.to_dict(),
            "execution_time_s": self.execution_time_s,
            "data": self.data,
        }


# JSON Schema definitions for tool argument validation
MOVE_COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {
            "type": "string",
            "enum": ["forward", "backward"],
            "description": "Direction to move"
        },
        "distance_m": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
            "description": "Distance to move in meters"
        }
    },
    "required": ["direction", "distance_m"]
}

TURN_COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "angle_deg": {
            "type": "number",
            "minimum": -360,
            "maximum": 360,
            "description": "Angle to turn in degrees (positive = clockwise)"
        }
    },
    "required": ["angle_deg"]
}

STOP_COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "Reason for stopping"
        },
        "emergency": {
            "type": "boolean",
            "description": "Whether this is an emergency stop"
        }
    }
}
