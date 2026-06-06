"""
Agent Interface - LangChain agent integration for Ranger robot using ROSA.

This module provides the RangerAgent class that integrates NASA JPL's ROSA
(Robot Operating System Agent) from the ros-technician-cli submodule with
Ranger-specific tools and prompts.

The agent:
- Uses ROSA as the base agent framework (from ros-technician-cli submodule)
- Extends ROSA with Ranger-specific tools (movement, status)
- Configures Ranger-specific prompts and persona
- Supports multiple LLM backends (OpenAI, Ollama, etc.)
- Streams responses and intermediate steps to the UI

Architecture:
    User Input -> RangerAgent (wraps ROSA) -> Ranger Tools + ROSA ROS2 Tools -> Robot
"""

import os
import sys
import logging
from typing import Optional, Any, AsyncIterator, Union, Literal
from enum import Enum

# Add ros-technician-cli submodule to path for ROSA imports
_submodule_src = os.path.join(os.path.dirname(__file__), '..', 'ros-technician-cli', 'src')
if os.path.exists(_submodule_src) and _submodule_src not in sys.path:
    sys.path.insert(0, os.path.abspath(_submodule_src))

# Import ROSA from the submodule
from rosa import ROSA, RobotSystemPrompts

# Import LangChain components for LLM creation
from langchain.agents import Tool
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback

# Import Ranger-specific components
from ranger_llm_ui.tools.all_tools import get_all_tools, initialize_all_tools
from ranger_llm_ui.tools.movement_tools import get_movement_tools
from ranger_llm_ui.tools.status_tools import get_status_tools
from ranger_llm_ui.ranger_prompts import get_ranger_prompts, RANGER_PROMPTS
from ranger_llm_ui.utils.logger import get_command_logger

logger = logging.getLogger(__name__)


_CLAUDE_CODE_SDK_PATCHED = False
_CLAUDE_ENV_STRIPPED = False
_LC_ANTHROPIC_PROXY_PATCHED = False


def _patch_langchain_anthropic_for_proxy():
    """
    CLIProxyAPI returns `context_management` as a plain dict in streaming
    `message_delta` events. langchain-anthropic 0.3.x calls `.model_dump()`
    on it unconditionally, raising AttributeError. Patch the event
    constructor to accept either a Pydantic model or a dict.
    """
    global _LC_ANTHROPIC_PROXY_PATCHED
    if _LC_ANTHROPIC_PROXY_PATCHED:
        return
    try:
        from langchain_anthropic import chat_models as _lcam
    except ImportError:
        return

    _orig_fn = _lcam._make_message_chunk_from_anthropic_event

    def _patched(event, *args, **kwargs):
        cm = getattr(event, "context_management", None)
        if cm is not None and not hasattr(cm, "model_dump"):
            try:
                event.context_management = None
            except Exception:
                pass
        return _orig_fn(event, *args, **kwargs)

    _lcam._make_message_chunk_from_anthropic_event = _patched
    _LC_ANTHROPIC_PROXY_PATCHED = True
    logger.info("Patched langchain-anthropic chunk builder to tolerate dict context_management from proxy")


def _strip_nested_claude_env():
    """
    Strip Claude Code environment variables that leak into this process when
    launched from inside a Claude Code terminal session, AND patch the
    claude-code-sdk subprocess transport to spawn the `claude` CLI with a
    minimal whitelisted env (mirroring `env -i`). Anything beyond the
    whitelist — including ROS LD_LIBRARY_PATH, AMENT_PREFIX_PATH, nested-AI
    markers — has been observed to cause the spawned claude CLI to exit 1
    with empty stderr when this process is launched from inside Claude Code.
    """
    global _CLAUDE_ENV_STRIPPED
    if _CLAUDE_ENV_STRIPPED:
        return

    leak_vars = [
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_EXECPATH",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_EFFORT",
        "AI_AGENT",
        "GIT_EDITOR",
    ]
    stripped = [v for v in leak_vars if v in os.environ]
    for v in stripped:
        os.environ.pop(v, None)
    if stripped:
        logger.info(f"Stripped leaked Claude Code env vars: {stripped}")

    # Patch SDK subprocess spawn to enforce minimal env for the `claude` CLI.
    try:
        from claude_code_sdk._internal.transport import subprocess_cli as _scli

        _orig_connect = _scli.SubprocessCLITransport.connect

        # Also patch anyio.open_process to redirect stderr to file for debug
        import anyio
        _orig_open_process = anyio.open_process

        async def _patched_open_process(cmd, **kwargs):
            try:
                cmd_str = str(cmd)
                if "claude" in cmd_str:
                    f = open("/tmp/claude_stderr_capture.log", "ab", buffering=0)
                    kwargs["stderr"] = f
                    # Dump full cmd to file for inspection
                    with open("/tmp/claude_cmd.log", "a") as cf:
                        import json as _json
                        cf.write(_json.dumps({
                            "cmd": cmd if isinstance(cmd, list) else [cmd],
                            "env_keys": sorted(kwargs.get('env', {}).keys()),
                            "env_full": dict(kwargs.get('env', {})),
                        }, indent=2, default=str) + "\n---\n")
                    logger.warning("[open_process] claude invocation logged to /tmp/claude_cmd.log")
            except Exception as e:
                logger.warning(f"open_process patch err: {e}")
            return await _orig_open_process(cmd, **kwargs)

        anyio.open_process = _patched_open_process

        async def _patched_connect(self):
            allow = {
                "HOME", "PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE",
                "SHELL", "TERM", "TMPDIR", "PWD",
                "NODE_PATH", "NPM_CONFIG_PREFIX",
                "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
            }
            clean_env = {k: v for k, v in os.environ.items() if k in allow}
            saved = dict(os.environ)
            os.environ.clear()
            os.environ.update(clean_env)
            try:
                return await _orig_connect(self)
            finally:
                os.environ.clear()
                os.environ.update(saved)

        _scli.SubprocessCLITransport.connect = _patched_connect
        logger.info("Patched SubprocessCLITransport.connect to use minimal env for claude CLI")
    except Exception as e:
        logger.warning(f"Could not patch SDK subprocess env: {e}")

    _CLAUDE_ENV_STRIPPED = True


def _patch_claude_code_sdk_parser():
    """
    Monkey-patch claude_code_sdk message parser to swallow unknown but
    informational frames (e.g. `rate_limit_event`) that the SDK version
    pinned here doesn't recognize. Without this, a benign rate-limit
    notification raises MessageParseError and aborts the chat turn.
    Upstream issue: claude-code-sdk has no case for `rate_limit_event`.
    """
    global _CLAUDE_CODE_SDK_PATCHED
    if _CLAUDE_CODE_SDK_PATCHED:
        return
    try:
        from claude_code_sdk._internal import message_parser as _mp
        from claude_code_sdk.types import SystemMessage
        from claude_code_sdk._errors import MessageParseError
    except ImportError:
        return

    _original_parse = _mp.parse_message
    # Message types known to be benign informational frames that older SDK
    # versions don't model. Coerce to a SystemMessage instead of raising.
    _ignorable_types = {"rate_limit_event"}

    def _patched(data):
        try:
            return _original_parse(data)
        except MessageParseError:
            if isinstance(data, dict) and data.get("type") in _ignorable_types:
                logger.debug(f"Coerced ignorable frame to SystemMessage: {data.get('type')}")
                return SystemMessage(subtype=data["type"], data=data)
            raise

    _mp.parse_message = _patched
    # Also patch the symbol re-exported into the client module.
    try:
        from claude_code_sdk._internal import client as _client
        _client.parse_message = _patched
    except ImportError:
        pass
    _CLAUDE_CODE_SDK_PATCHED = True
    logger.info("Patched claude_code_sdk.parse_message to ignore rate_limit_event frames")


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    CLAUDE_CODE = "claude_code"
    CLAUDE_PROXY = "claude_proxy"


def create_llm(
    provider: LLMProvider = LLMProvider.OPENAI,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    streaming: bool = True,
    **kwargs,
):
    """
    Create an LLM instance based on the provider.

    Args:
        provider: LLM provider (openai, ollama, anthropic)
        model_name: Model name (defaults to provider's default)
        temperature: Temperature for generation
        streaming: Enable streaming responses
        **kwargs: Additional provider-specific arguments

    Returns:
        LLM instance compatible with ROSA
    """
    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        max_tokens_env = os.getenv("LLM_MAX_TOKENS")
        max_tokens = int(max_tokens_env) if max_tokens_env else None

        return ChatOpenAI(
            # Default to a cheaper, widely available model. Override with LLM_MODEL.
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
            api_key=api_key,
            streaming=streaming,
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            **{k: v for k, v in kwargs.items() if k not in ["api_key"]},
        )

    elif provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        base_url = kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        return ChatOllama(
            model=model_name or "llama2",
            temperature=temperature,
            base_url=base_url,
            **{k: v for k, v in kwargs.items() if k not in ["base_url"]},
        )

    elif provider == LLMProvider.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Install langchain-anthropic: pip install langchain-anthropic")

        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")

        return ChatAnthropic(
            model=model_name or "claude-3-sonnet-20240229",
            temperature=temperature,
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k not in ["api_key"]},
        )

    elif provider == LLMProvider.CLAUDE_CODE:
        try:
            from langchain_claude_code import ChatClaudeCode
        except ImportError:
            raise ImportError(
                "Install langchain-claude-code-cli: pip install langchain-claude-code-cli. "
                "Also requires Claude Code CLI: npm install -g @anthropic-ai/claude-code"
            )

        _patch_claude_code_sdk_parser()
        _strip_nested_claude_env()

        # Auth resolution (in order of precedence):
        # 1. CLAUDE_CODE_OAUTH_TOKEN env var (read by claude-code-sdk subprocess)
        # 2. Active `claude login` session (CLI uses its stored credentials)
        # 3. ANTHROPIC_API_KEY env var (pay-per-token billing)
        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")

        # Short aliases → full model IDs (langchain_claude_code expects full names)
        model_aliases = {
            "opus": "claude-opus-4-8",
            "opus-4.8": "claude-opus-4-8",
            "opus-4.7": "claude-opus-4-7",
            "sonnet": "claude-sonnet-4-6",
            "sonnet-4.6": "claude-sonnet-4-6",
            "sonnet-4": "claude-sonnet-4-20250514",
            "haiku": "claude-haiku-4-5-20251001",
            "haiku-4.5": "claude-haiku-4-5-20251001",
        }
        resolved_model = model_aliases.get(
            (model_name or "sonnet-4.6").lower(),
            model_name or "claude-sonnet-4-6",
        )

        chat_kwargs = {
            "model": resolved_model,
            "streaming": streaming,
        }
        if temperature is not None:
            chat_kwargs["temperature"] = temperature
        if api_key:
            chat_kwargs["api_key"] = api_key

        cli_path = os.getenv("CLAUDE_CODE_CLI_PATH")
        if cli_path:
            chat_kwargs["cli_path"] = cli_path

        chat_kwargs.update({
            k: v for k, v in kwargs.items() if k not in ["api_key", "oauth_token"]
        })

        return ChatClaudeCode(**chat_kwargs)

    elif provider == LLMProvider.CLAUDE_PROXY:
        # Route through a local CLIProxyAPI instance that wraps the user's
        # Claude Pro/Max subscription OAuth as an Anthropic-compatible HTTP
        # endpoint. Tool calling works end-to-end via ChatAnthropic.bind_tools.
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Install langchain-anthropic: pip install langchain-anthropic")

        _patch_langchain_anthropic_for_proxy()

        base_url = (
            kwargs.get("base_url")
            or os.getenv("CLAUDE_PROXY_BASE_URL")
            or "http://127.0.0.1:8317"
        )
        api_key = (
            kwargs.get("api_key")
            or os.getenv("CLAUDE_PROXY_API_KEY")
            or "ranger-local-key"
        )

        model_aliases = {
            "opus": "claude-opus-4-8",
            "opus-4.8": "claude-opus-4-8",
            "opus-4.7": "claude-opus-4-7",
            "sonnet": "claude-sonnet-4-6",
            "sonnet-4.6": "claude-sonnet-4-6",
            "sonnet-4.5": "claude-sonnet-4-5-20250929",
            "sonnet-4": "claude-sonnet-4-20250514",
            "haiku": "claude-haiku-4-5-20251001",
            "haiku-4.5": "claude-haiku-4-5-20251001",
        }
        resolved_model = model_aliases.get(
            (model_name or "sonnet-4.6").lower(),
            model_name or "claude-sonnet-4-6",
        )

        return ChatAnthropic(
            model=resolved_model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            streaming=streaming,
            **{k: v for k, v in kwargs.items() if k not in ["api_key", "base_url"]},
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


class RangerAgent:
    """
    LangChain agent for the Ranger robot, built on NASA JPL's ROSA.

    This agent wraps ROSA from the ros-technician-cli submodule and extends it
    with Ranger-specific tools for movement and status queries. It inherits
    ROSA's ROS2 introspection capabilities while adding robot control.

    The integration follows the design document specification:
    - Uses ROSA as the base agent (from submodule)
    - Registers Ranger-specific tools with ROSA
    - Configures Ranger-specific prompts
    - Supports streaming responses for the Gradio UI
    """

    def __init__(
        self,
        llm=None,
        provider: LLMProvider = LLMProvider.OPENAI,
        model_name: Optional[str] = None,
        ros_node: Optional[Any] = None,
        verbose: bool = False,
        streaming: bool = True,
        debug_mode: bool = False,
    ):
        """
        Initialize the Ranger agent.

        Args:
            llm: Pre-configured LLM instance (optional)
            provider: LLM provider if llm not provided
            model_name: Model name if llm not provided
            ros_node: ROS 2 node for tool initialization
            verbose: Enable verbose logging
            streaming: Enable streaming responses (default: True)
            debug_mode: When True, skip Ranger-specific prompts and use only
                        ROSA base system prompts (useful for debugging)
        """
        # Initialize Ranger-specific tools with ROS node
        self._initialize_ranger_tools(ros_node)

        # Create or use provided LLM
        if llm is None:
            llm = create_llm(
                provider=provider,
                model_name=model_name,
                streaming=streaming,
            )

        # Get Ranger-specific tools as LangChain tools
        ranger_tools = self._wrap_tools(get_all_tools())

        # Get Ranger-specific prompts (None in debug mode → ROSA base prompts only)
        ranger_prompts = None if debug_mode else get_ranger_prompts()
        self.debug_mode = debug_mode

        max_iterations = int(os.getenv("ROSA_MAX_ITERATIONS", "15"))
        self._max_history_messages = int(os.getenv("ROSA_MAX_HISTORY_MESSAGES", "20"))
        agent_verbose = os.getenv("ROSA_VERBOSE", "false").lower() in ("true", "1", "yes")

        # Create ROSA instance with Ranger tools and prompts
        # ROSA handles all the agent logic, tool binding, and execution
        self._rosa = ROSA(
            ros_version=2,  # ROS 2
            llm=llm,
            tools=ranger_tools,  # Add Ranger-specific tools
            prompts=ranger_prompts,  # Ranger-specific prompts
            verbose=verbose or agent_verbose,
            streaming=streaming,
            accumulate_chat_history=True,
            max_iterations=max_iterations,
            return_intermediate_steps=False,
        )

        # Command logger for tracking
        self._logger = get_command_logger()

        logger.info(f"RangerAgent initialized with ROSA (ros-technician-cli submodule)")
        logger.info(f"Ranger tools: {[t.name for t in ranger_tools]}")
        if debug_mode:
            logger.info("DEBUG MODE: Ranger-specific prompts disabled. Using ROSA base prompts only.")

    def _wrap_tools(self, tools: list[Any]) -> list[Any]:
        """
        ROSA's tool registry expects LangChain Tools. For tools with structured
        args_schema (like CameraImageInput), we return them as-is since BaseTool
        handles structured inputs properly. For simple tools, we can wrap them.
        """

        wrapped: list[Any] = []

        for base_tool in tools:
            # Check if tool has a structured args_schema (Pydantic model)
            has_structured_schema = hasattr(base_tool, "args_schema") and base_tool.args_schema is not None

            if has_structured_schema:
                # Tool already has proper schema, don't wrap it - return as BaseTool
                # Modern LangChain agents can handle BaseTool directly
                logger.debug(f"Keeping structured tool as-is: {base_tool.name}")
                wrapped.append(base_tool)
            else:
                # Simple tool without schema, wrap in old-style Tool
                logger.debug(f"Wrapping simple tool: {base_tool.name}")
                # Create a closure that captures the tool instance
                def make_wrapper(tool_instance):
                    def _run_wrapped(*args, **kwargs):
                        # Handle both calling conventions:
                        # 1. Called with a dict as single positional arg: func({"key": "value"})
                        # 2. Called with kwargs: func(key="value")
                        if args and len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                            # Case 1: single dict argument
                            return tool_instance._run(**args[0])
                        else:
                            # Case 2: keyword arguments
                            return tool_instance._run(**kwargs)
                    return _run_wrapped

                wrapped.append(
                    Tool(
                        name=base_tool.name,
                        description=base_tool.description,
                        func=make_wrapper(base_tool),
                    )
                )

        return wrapped

    def _trim_chat_history(self):
        """Trim chat history to prevent context length overflow."""
        if self._max_history_messages <= 0:
            return
        history = self._rosa.chat_history
        if len(history) > self._max_history_messages:
            # Keep only the most recent messages
            del history[: len(history) - self._max_history_messages]
            logger.info(f"Trimmed chat history to {len(history)} messages")

    def _initialize_ranger_tools(self, ros_node: Optional[Any]):
        """Initialize Ranger tools with the ROS node."""
        from ranger_llm_ui.tools.movement_tools import initialize_ros_interface
        from ranger_llm_ui.tools.status_tools import initialize_status_interface
        from ranger_llm_ui.tools.camera_tools import initialize_camera_interface
        from ranger_llm_ui.tools.manipulation_tools import (
            initialize_manipulation_interface,
        )

        initialize_ros_interface(ros_node)
        initialize_status_interface(ros_node)
        initialize_camera_interface(ros_node)
        initialize_manipulation_interface(ros_node)

    def invoke(self, user_input: str) -> dict:
        """
        Process a user command synchronously.

        Args:
            user_input: Natural language command from user

        Returns:
            Dictionary with 'output' and optionally 'intermediate_steps'
        """
        # Log user input
        self._logger.log_conversation(role="user", content=user_input)

        # Invoke ROSA agent
        try:
            self._trim_chat_history()
            with get_openai_callback() as cb:
                result = self._rosa._ROSA__executor.invoke(  # type: ignore[attr-defined]
                    {"input": user_input, "chat_history": self._rosa.chat_history}
                )

            output = result.get("output", "")
            usage = {
                "prompt_tokens": getattr(cb, "prompt_tokens", 0),
                "completion_tokens": getattr(cb, "completion_tokens", 0),
                "total_tokens": getattr(cb, "total_tokens", 0),
                "total_cost_usd": getattr(cb, "total_cost", 0.0),
                "successful_requests": getattr(cb, "successful_requests", 0),
            }

            if getattr(self._rosa, "_ROSA__accumulate_chat_history", True):  # type: ignore[attr-defined]
                self._rosa.chat_history.extend(
                    [HumanMessage(content=user_input), AIMessage(content=output)]
                )

            # Log assistant response
            self._logger.log_conversation(role="assistant", content=output)

            return {
                "output": output,
                "intermediate_steps": result.get("intermediate_steps", []),
                "usage": usage,
            }

        except Exception as e:
            logger.error(f"Agent error: {e}")
            error_msg = f"I encountered an error: {str(e)}"
            self._logger.log_conversation(role="assistant", content=error_msg)
            return {"output": error_msg, "intermediate_steps": [], "usage": {}}

    async def astream(self, user_input: str) -> AsyncIterator[dict]:
        """
        Process a user command with streaming response.

        Args:
            user_input: Natural language command from user

        Yields:
            Dictionaries with intermediate steps and final output
        """
        # Log user input
        self._logger.log_conversation(role="user", content=user_input)

        try:
            self._trim_chat_history()
            # Use ROSA's async streaming
            async for event in self._rosa.astream(user_input):
                event_type = event.get("type")

                if event_type == "token":
                    yield {"type": "token", "content": event.get("content", "")}

                elif event_type == "tool_start":
                    yield {
                        "type": "tool_start",
                        "tool": event.get("name"),
                        "input": event.get("input"),
                    }

                elif event_type == "tool_end":
                    yield {
                        "type": "tool_end",
                        "output": event.get("output"),
                    }

                elif event_type == "final":
                    final_output = event.get("content", "")
                    self._logger.log_conversation(role="assistant", content=final_output)
                    yield {"type": "final", "output": final_output}

                elif event_type == "error":
                    error_msg = event.get("content", "Unknown error")
                    self._logger.log_conversation(role="assistant", content=error_msg)
                    yield {"type": "error", "error": error_msg}

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_msg = f"I encountered an error: {str(e)}"
            self._logger.log_conversation(role="assistant", content=error_msg)
            yield {"type": "error", "error": error_msg}

    def clear_history(self):
        """Clear chat history."""
        self._rosa.clear_chat()
        logger.info("Chat history cleared")

    def get_tool_names(self) -> list[str]:
        """Get list of available tool names."""
        # ROSA tools include both Ranger tools and ROS2 introspection tools
        return [tool.name for tool in self._rosa._ROSA__tools.get_tools()]

    def get_chat_history(self) -> list[dict]:
        """Get chat history as list of dicts."""
        history = []
        for msg in self._rosa.chat_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history


class SimpleAgent:
    """
    Simplified agent for basic testing without full LangChain/ROSA setup.

    This agent provides basic command parsing without requiring an LLM,
    useful for testing the tool infrastructure.
    """

    def __init__(self, ros_node: Optional[Any] = None):
        """Initialize the simple agent."""
        self.tools = initialize_all_tools(ros_node)
        self.tool_map = {tool.name.lower(): tool for tool in self.tools}
        self.logger = get_command_logger()

    def invoke(self, user_input: str) -> dict:
        """
        Process a simple command.

        Supports basic commands like:
        - "move forward 1" -> MoveForward(1.0)
        - "turn left 90" -> TurnAngle(-90)
        - "stop" -> StopRobot()
        - "battery" -> BatteryStatus()
        """
        self.logger.log_conversation(role="user", content=user_input)

        user_input_lower = user_input.lower().strip()

        # Parse simple commands
        if "stop" in user_input_lower:
            result = self.tool_map["stoprobot"].run({})
        elif "battery" in user_input_lower:
            result = self.tool_map["batterystatus"].run({})
        elif "status" in user_input_lower or "health" in user_input_lower:
            result = self.tool_map["systemhealth"].run({})
        elif "forward" in user_input_lower:
            # Extract distance
            parts = user_input_lower.split()
            distance = 1.0
            for i, part in enumerate(parts):
                if part == "forward" and i + 1 < len(parts):
                    try:
                        distance = float(parts[i + 1])
                    except ValueError:
                        pass
            result = self.tool_map["moveforward"].run({"distance_m": distance})
        elif "backward" in user_input_lower or "back" in user_input_lower:
            parts = user_input_lower.split()
            distance = 1.0
            for i, part in enumerate(parts):
                if part in ["backward", "back"] and i + 1 < len(parts):
                    try:
                        distance = float(parts[i + 1])
                    except ValueError:
                        pass
            result = self.tool_map["movebackward"].run({"distance_m": distance})
        elif "turn" in user_input_lower or "rotate" in user_input_lower:
            parts = user_input_lower.split()
            angle = 90.0
            if "left" in user_input_lower:
                angle = -90.0
            for part in parts:
                if part.replace("-", "").replace(".", "").isdigit():
                    angle = float(part)
                    break
            result = self.tool_map["turnangle"].run({"angle_deg": angle})
        elif "camera" in user_input_lower or "image" in user_input_lower:
            result = self.tool_map["getcameraimage"].run({})
        else:
            result = f"I don't understand '{user_input}'. Try: move forward 1, turn left 90, stop, or battery"

        self.logger.log_conversation(role="assistant", content=result)
        return {"output": result, "intermediate_steps": []}


def create_agent(
    provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
    model_name: Optional[str] = None,
    ros_node: Optional[Any] = None,
    simple_mode: bool = False,
    debug_mode: bool = False,
    **kwargs,
) -> Union[RangerAgent, SimpleAgent]:
    """
    Factory function to create a Ranger agent.

    Args:
        provider: LLM provider ("openai", "ollama", "anthropic") or LLMProvider enum
        model_name: Model name (optional)
        ros_node: ROS 2 node for tool initialization
        simple_mode: Use SimpleAgent for testing without LLM
        debug_mode: Skip Ranger-specific prompts, use only ROSA base prompts
        **kwargs: Additional arguments for LLM creation

    Returns:
        RangerAgent or SimpleAgent instance
    """
    if simple_mode:
        return SimpleAgent(ros_node=ros_node)

    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())

    return RangerAgent(
        provider=provider,
        model_name=model_name,
        ros_node=ros_node,
        debug_mode=debug_mode,
        **kwargs,
    )
