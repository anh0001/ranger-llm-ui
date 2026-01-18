"""
Safety Module - Safety checks and interlocks for robot operations.

This module provides safety constraints at multiple levels:
- Velocity and acceleration limits
- Distance validation
- Confirmation requirements for high-impact actions
- Emergency stop functionality
"""

from ranger_llm_ui.safety.guard import (
    SafetyGuard,
    SafetyConfig,
    validate_velocity,
    validate_distance,
    requires_confirmation,
)

__all__ = [
    "SafetyGuard",
    "SafetyConfig",
    "validate_velocity",
    "validate_distance",
    "requires_confirmation",
]
