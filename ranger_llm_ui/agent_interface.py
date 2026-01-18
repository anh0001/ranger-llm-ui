"""
Agent Interface - LangChain agent integration for Ranger robot.

This module provides the RangerAgent class that integrates the LangChain
agent framework with Ranger-specific tools. It supports multiple LLM
backends (OpenAI, Ollama, etc.) through a provider abstraction.

The agent:
- Interprets natural language commands from users
- Selects appropriate tools to fulfill requests
- Executes robot actions through the tool registry
- Streams responses and intermediate steps to the UI
"""

import os
import logging
from typing import Optional, Any, AsyncIterator, Union
from enum import Enum

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.tools import BaseTool

from ranger_llm_ui.tools.all_tools import get_all_tools, initialize_all_tools
from ranger_llm_ui.utils.logger import get_command_logger

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


# Default system prompt for the Ranger agent
RANGER_SYSTEM_PROMPT = """You are a helpful robot assistant for the Ranger garden robot.
You help operators control the robot using natural language commands.

You have access to the following tools:

{tools}

IMPORTANT SAFETY RULES:
1. Always prioritize safety. If a command seems dangerous, ask for confirmation.
2. Only use the tools provided. Do not invent new capabilities.
3. If you cannot fulfill a request with the available tools, explain what you can do instead.
4. For large movements (>2 meters), mention that this is a significant distance.
5. If the user says "stop" or "emergency", immediately use the StopRobot tool.

When responding:
- Be concise and clear
- Report the result of any action you take
- If an action fails, explain what went wrong
- Proactively mention relevant status information (like battery level after movement)

Use the following format:

Question: the input question or command you must respond to
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer or response to the user

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


def create_llm(
    provider: LLMProvider = LLMProvider.OPENAI,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    """
    Create an LLM instance based on the provider.

    Args:
        provider: LLM provider (openai, ollama, anthropic)
        model_name: Model name (defaults to provider's default)
        temperature: Temperature for generation
        **kwargs: Additional provider-specific arguments

    Returns:
        BaseChatModel instance
    """
    if provider == LLMProvider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Install langchain-openai: pip install langchain-openai")

        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        return ChatOpenAI(
            model=model_name or "gpt-4",
            temperature=temperature,
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k != "api_key"},
        )

    elif provider == LLMProvider.OLLAMA:
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            raise ImportError("Install langchain-community: pip install langchain-community")

        base_url = kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        return ChatOllama(
            model=model_name or "llama2",
            temperature=temperature,
            base_url=base_url,
            **{k: v for k, v in kwargs.items() if k != "base_url"},
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
            **{k: v for k, v in kwargs.items() if k != "api_key"},
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


class RangerAgent:
    """
    LangChain agent for the Ranger robot.

    This agent interprets natural language commands and executes robot
    actions through the tool registry. It supports streaming responses
    and multiple LLM backends.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        provider: LLMProvider = LLMProvider.OPENAI,
        model_name: Optional[str] = None,
        tools: Optional[list[BaseTool]] = None,
        ros_node: Optional[Any] = None,
        verbose: bool = False,
    ):
        """
        Initialize the Ranger agent.

        Args:
            llm: Pre-configured LLM instance (optional)
            provider: LLM provider if llm not provided
            model_name: Model name if llm not provided
            tools: Custom tools list (default: all Ranger tools)
            ros_node: ROS 2 node for tool initialization
            verbose: Enable verbose logging
        """
        # Initialize tools with ROS node
        self.tools = tools or initialize_all_tools(ros_node)

        # Create or use provided LLM
        if llm is not None:
            self.llm = llm
        else:
            self.llm = create_llm(provider=provider, model_name=model_name)

        # Create prompt
        self.prompt = PromptTemplate.from_template(RANGER_SYSTEM_PROMPT)

        # Create agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )

        # Create executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=verbose,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=True,
        )

        # Logger for conversation tracking
        self.logger = get_command_logger()

        # Chat history
        self.chat_history: list[BaseMessage] = []

        logger.info(f"RangerAgent initialized with {len(self.tools)} tools")

    def invoke(self, user_input: str) -> dict:
        """
        Process a user command synchronously.

        Args:
            user_input: Natural language command from user

        Returns:
            Dictionary with 'output' and 'intermediate_steps'
        """
        # Log user input
        self.logger.log_conversation(role="user", content=user_input)
        self.chat_history.append(HumanMessage(content=user_input))

        # Invoke agent
        try:
            result = self.executor.invoke({
                "input": user_input,
                "chat_history": self.chat_history,
            })

            # Log assistant response
            output = result.get("output", "")
            self.logger.log_conversation(role="assistant", content=output)
            self.chat_history.append(AIMessage(content=output))

            return result

        except Exception as e:
            logger.error(f"Agent error: {e}")
            error_msg = f"I encountered an error: {str(e)}"
            self.logger.log_conversation(role="assistant", content=error_msg)
            return {"output": error_msg, "intermediate_steps": []}

    async def astream(self, user_input: str) -> AsyncIterator[dict]:
        """
        Process a user command with streaming response.

        Args:
            user_input: Natural language command from user

        Yields:
            Dictionaries with intermediate steps and final output
        """
        # Log user input
        self.logger.log_conversation(role="user", content=user_input)
        self.chat_history.append(HumanMessage(content=user_input))

        try:
            async for event in self.executor.astream_events(
                {"input": user_input, "chat_history": self.chat_history},
                version="v1",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    # Streaming token from LLM
                    content = event["data"]["chunk"].content
                    if content:
                        yield {"type": "token", "content": content}

                elif kind == "on_tool_start":
                    # Tool is starting
                    tool_name = event["name"]
                    tool_input = event["data"].get("input", {})
                    yield {
                        "type": "tool_start",
                        "tool": tool_name,
                        "input": tool_input,
                    }

                elif kind == "on_tool_end":
                    # Tool completed
                    tool_output = event["data"].get("output", "")
                    yield {
                        "type": "tool_end",
                        "output": tool_output,
                    }

                elif kind == "on_chain_end":
                    # Final output
                    if "output" in event["data"]:
                        output = event["data"]["output"]
                        if isinstance(output, dict) and "output" in output:
                            final_output = output["output"]
                            self.logger.log_conversation(role="assistant", content=final_output)
                            self.chat_history.append(AIMessage(content=final_output))
                            yield {"type": "final", "output": final_output}

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_msg = f"I encountered an error: {str(e)}"
            self.logger.log_conversation(role="assistant", content=error_msg)
            yield {"type": "error", "error": error_msg}

    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        logger.info("Chat history cleared")

    def get_tool_names(self) -> list[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools]

    def get_chat_history(self) -> list[dict]:
        """Get chat history as list of dicts."""
        history = []
        for msg in self.chat_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history


class SimpleAgent:
    """
    Simplified agent for basic testing without full LangChain setup.

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

        user_input = user_input.lower().strip()

        # Parse simple commands
        if "stop" in user_input:
            result = self.tool_map["stoprobot"].run({})
        elif "battery" in user_input:
            result = self.tool_map["batterystatus"].run({})
        elif "status" in user_input or "health" in user_input:
            result = self.tool_map["systemhealth"].run({})
        elif "forward" in user_input:
            # Extract distance
            parts = user_input.split()
            distance = 1.0
            for i, part in enumerate(parts):
                if part == "forward" and i + 1 < len(parts):
                    try:
                        distance = float(parts[i + 1])
                    except ValueError:
                        pass
            result = self.tool_map["moveforward"].run({"distance_m": distance})
        elif "backward" in user_input or "back" in user_input:
            parts = user_input.split()
            distance = 1.0
            for i, part in enumerate(parts):
                if part in ["backward", "back"] and i + 1 < len(parts):
                    try:
                        distance = float(parts[i + 1])
                    except ValueError:
                        pass
            result = self.tool_map["movebackward"].run({"distance_m": distance})
        elif "turn" in user_input or "rotate" in user_input:
            parts = user_input.split()
            angle = 90.0
            if "left" in user_input:
                angle = -90.0
            for i, part in enumerate(parts):
                if part.replace("-", "").replace(".", "").isdigit():
                    angle = float(part)
                    break
            result = self.tool_map["turnangle"].run({"angle_deg": angle})
        else:
            result = f"I don't understand '{user_input}'. Try: move forward 1, turn left 90, stop, or battery"

        self.logger.log_conversation(role="assistant", content=result)
        return {"output": result, "intermediate_steps": []}


def create_agent(
    provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
    model_name: Optional[str] = None,
    ros_node: Optional[Any] = None,
    simple_mode: bool = False,
    **kwargs,
) -> Union[RangerAgent, SimpleAgent]:
    """
    Factory function to create a Ranger agent.

    Args:
        provider: LLM provider ("openai", "ollama", "anthropic") or LLMProvider enum
        model_name: Model name (optional)
        ros_node: ROS 2 node for tool initialization
        simple_mode: Use SimpleAgent for testing without LLM
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
        **kwargs,
    )
