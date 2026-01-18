"""
Structured Logging - Command logging and history tracking.

All agent decisions and tool calls generate structured logs. Each time the
agent invokes a tool, we log a JSON command record (including timestamp,
tool name, parameters, and result).
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from collections import deque
import threading

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a tool invocation."""
    timestamp: str
    tool_name: str
    parameters: dict
    result: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    session_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ConversationRecord:
    """Record of a conversation turn."""
    timestamp: str
    role: str  # "user" or "assistant"
    content: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    session_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "session_id": self.session_id,
        }


class CommandLogger:
    """
    Logger for robot commands and agent interactions.

    Provides structured logging to file and maintains an in-memory
    history for UI display and debugging.
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        max_history_size: int = 1000,
        session_id: Optional[str] = None,
    ):
        self.log_dir = log_dir or Path.home() / ".ranger_llm_ui" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"session_{self.session_id}.jsonl"

        self._history: deque[ToolCallRecord] = deque(maxlen=max_history_size)
        self._conversation_history: deque[ConversationRecord] = deque(
            maxlen=max_history_size
        )
        self._lock = threading.Lock()

        logger.info(f"CommandLogger initialized. Log file: {self.log_file}")

    def log_tool_call(
        self,
        tool_name: str,
        parameters: dict,
        result: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        execution_time_ms: float = 0.0,
    ) -> ToolCallRecord:
        """
        Log a tool invocation.

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            result: Tool result string
            success: Whether the tool succeeded
            error: Error message if failed
            execution_time_ms: Execution time in milliseconds

        Returns:
            The created ToolCallRecord
        """
        record = ToolCallRecord(
            timestamp=datetime.now().isoformat(),
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            success=success,
            error=error,
            execution_time_ms=execution_time_ms,
            session_id=self.session_id,
        )

        with self._lock:
            self._history.append(record)
            self._write_to_file(record.to_dict())

        # Also log to standard logger
        log_level = logging.INFO if success else logging.ERROR
        logger.log(
            log_level,
            f"Tool: {tool_name} | Params: {parameters} | "
            f"Success: {success} | Time: {execution_time_ms:.1f}ms",
        )

        return record

    def log_conversation(
        self,
        role: str,
        content: str,
        tool_calls: Optional[list[ToolCallRecord]] = None,
    ) -> ConversationRecord:
        """
        Log a conversation turn.

        Args:
            role: "user" or "assistant"
            content: Message content
            tool_calls: List of tool calls made during this turn

        Returns:
            The created ConversationRecord
        """
        record = ConversationRecord(
            timestamp=datetime.now().isoformat(),
            role=role,
            content=content,
            tool_calls=tool_calls or [],
            session_id=self.session_id,
        )

        with self._lock:
            self._conversation_history.append(record)
            self._write_to_file({"conversation": record.to_dict()})

        return record

    def _write_to_file(self, data: dict):
        """Write a record to the log file (JSON Lines format)."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    def get_tool_history(self, limit: int = 50) -> list[ToolCallRecord]:
        """Get recent tool call history."""
        with self._lock:
            return list(self._history)[-limit:]

    def get_conversation_history(self, limit: int = 50) -> list[ConversationRecord]:
        """Get recent conversation history."""
        with self._lock:
            return list(self._conversation_history)[-limit:]

    def get_history_summary(self) -> dict:
        """Get a summary of the current session."""
        with self._lock:
            tool_counts: dict[str, int] = {}
            success_count = 0
            error_count = 0

            for record in self._history:
                tool_counts[record.tool_name] = tool_counts.get(record.tool_name, 0) + 1
                if record.success:
                    success_count += 1
                else:
                    error_count += 1

            return {
                "session_id": self.session_id,
                "total_tool_calls": len(self._history),
                "successful_calls": success_count,
                "failed_calls": error_count,
                "tool_usage": tool_counts,
                "conversation_turns": len(self._conversation_history),
            }

    def export_session(self, output_path: Optional[Path] = None) -> Path:
        """
        Export the current session to a JSON file.

        Args:
            output_path: Path to export to (defaults to log_dir)

        Returns:
            Path to the exported file
        """
        output_path = output_path or (
            self.log_dir / f"export_{self.session_id}.json"
        )

        with self._lock:
            export_data = {
                "session_id": self.session_id,
                "export_timestamp": datetime.now().isoformat(),
                "summary": self.get_history_summary(),
                "tool_calls": [r.to_dict() for r in self._history],
                "conversations": [r.to_dict() for r in self._conversation_history],
            }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Session exported to {output_path}")
        return output_path


# Module-level singleton and convenience functions
_default_logger: Optional[CommandLogger] = None


def get_command_logger() -> CommandLogger:
    """Get or create the default command logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = CommandLogger()
    return _default_logger


def log_tool_call(
    tool_name: str,
    parameters: dict,
    result: Optional[str] = None,
    success: bool = True,
    error: Optional[str] = None,
    execution_time_ms: float = 0.0,
) -> ToolCallRecord:
    """Log a tool call using the default logger."""
    return get_command_logger().log_tool_call(
        tool_name=tool_name,
        parameters=parameters,
        result=result,
        success=success,
        error=error,
        execution_time_ms=execution_time_ms,
    )


def get_command_history(limit: int = 50) -> list[ToolCallRecord]:
    """Get command history from the default logger."""
    return get_command_logger().get_tool_history(limit)


def setup_logging(
    level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_dir: Optional[Path] = None,
):
    """
    Set up logging configuration for the package.

    Args:
        level: Logging level
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_dir: Directory for log files
    """
    log_dir = log_dir or Path.home() / ".ranger_llm_ui" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = []

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(console_handler)

    if log_to_file:
        file_handler = logging.FileHandler(
            log_dir / f"ranger_llm_ui_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    # Configure root logger for the package
    pkg_logger = logging.getLogger("ranger_llm_ui")
    pkg_logger.setLevel(level)
    for handler in handlers:
        pkg_logger.addHandler(handler)
