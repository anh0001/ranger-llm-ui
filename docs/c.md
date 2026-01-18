# CLAUDE.md - Ranger LLM UI Project Guide

This document helps AI assistants (like Claude) understand the Ranger LLM UI codebase.

## Project Overview

**Ranger LLM UI** is an LLM-driven natural language operator interface for the Ranger garden robot. It allows operators to control a ROS 2 robot using conversational commands through a web-based chat interface.

**Key Features:**
- Natural language robot control via LangChain agents
- Gradio-based web chat UI with streaming responses
- Integration with NASA JPL's ROSA (Robot Operating System Agent)
- Multiple LLM backend support (OpenAI, Ollama, Anthropic)
- Safety-first design with velocity limits and emergency stop
- Native ROS 2 Humble integration

## Architecture

```
User (Web UI) → Gradio Interface → RangerAgent (wraps ROSA) → Ranger Tools → ROS 2 → Robot
                                          ↓
                                    ROSA ROS2 Tools
```

**Key Components:**
1. **Gradio UI** ([ui_node.py](ranger_llm_ui/ui_node.py)) - Web interface for chat and manual controls
2. **RangerAgent** ([agent_interface.py](ranger_llm_ui/agent_interface.py:122)) - LangChain agent wrapping ROSA
3. **Tools** ([tools/](ranger_llm_ui/tools/)) - LangChain tools for robot control and status
4. **ROSA** (submodule: [ros-technician-cli/](ros-technician-cli/)) - Base agent framework from NASA JPL
5. **ROS 2 Interface** - Native ROS 2 node for robot communication

## File Structure

```
ranger-llm-ui/
├── ranger_llm_ui/              # Main Python package
│   ├── ui_node.py              # Entry point: Gradio UI + ROS 2 node
│   ├── agent_interface.py      # RangerAgent class (wraps ROSA)
│   ├── ranger_prompts.py       # Ranger-specific prompts for the agent
│   ├── tools/
│   │   ├── movement_tools.py   # Movement: MoveForward, TurnAngle, etc.
│   │   ├── status_tools.py     # Status: BatteryStatus, SystemHealth, etc.
│   │   └── all_tools.py        # Tool registry and initialization
│   ├── schemas/
│   │   └── commands.py         # Dataclasses for ROS commands
│   ├── safety/
│   │   └── guard.py            # Safety validators and constraints
│   └── utils/
│       └── logger.py           # Logging utilities
├── ros-technician-cli/         # ROSA submodule (NASA JPL)
│   └── src/rosa/               # ROSA base agent framework
├── ros2_numpy/                 # NumPy conversions for ROS 2 (submodule)
├── config/
│   └── default_config.yaml     # Configuration: LLM, safety limits, topics
├── launch/
│   └── ranger_llm_ui.launch.py # ROS 2 launch file
├── tests/                      # pytest test suite
└── setup.py                    # Python/ROS 2 package setup
```

## Key Concepts

### 1. ROSA Integration

The project uses NASA JPL's **ROSA** (Robot Operating System Agent) as its foundation. ROSA provides:
- LangChain agent framework
- ROS 2 introspection tools (list nodes/topics, echo messages, etc.)
- Prompt management via `RobotSystemPrompts`

**RangerAgent extends ROSA** by:
- Adding Ranger-specific movement and status tools
- Configuring Ranger robot prompts and persona
- Wrapping ROSA's streaming interface for Gradio UI

See [agent_interface.py](ranger_llm_ui/agent_interface.py) for the integration.

### 2. Tool System

Tools are LangChain `BaseTool` subclasses that the agent can invoke. Tools are organized by category:

**Movement Tools** ([movement_tools.py](ranger_llm_ui/tools/movement_tools.py)):
- `MoveForwardTool` - Move forward by distance (meters)
- `MoveBackwardTool` - Move backward by distance
- `TurnAngleTool` - Rotate in place (degrees)
- `StopRobotTool` - Emergency stop

**Status Tools** ([status_tools.py](ranger_llm_ui/tools/status_tools.py)):
- `BatteryStatusTool` - Get battery level
- `SystemHealthTool` - Check system health
- `GetOdometryTool` - Get current pose
- `ListNodesTool` / `ListTopicsTool` - ROS 2 diagnostics

**Tool Registry** ([all_tools.py](ranger_llm_ui/tools/all_tools.py)):
- Central registry for all tools
- Tool initialization with ROS node
- Tool usage tracking

### 3. Prompts and Persona

The agent's behavior is controlled by prompts defined in [ranger_prompts.py](ranger_llm_ui/ranger_prompts.py).

**Prompt Structure** (using ROSA's `RobotSystemPrompts`):
- `embodiment_and_persona` - Robot identity ("You are Ranger, a garden robot...")
- `critical_instructions` - Safety rules (emergency stop, movement limits)
- `constraints_and_guardrails` - What the agent must/must not do
- `about_your_capabilities` - Available tools and actions
- `about_your_environment` - Operating context (garden, ROS topics)
- `mission_and_objectives` - Primary goals

**Important:** The agent speaks AS the robot (first person: "I", "my").

### 4. Safety System

Safety is enforced at multiple levels:

**Configuration** ([default_config.yaml](config/default_config.yaml)):
- `max_linear_velocity: 0.5 m/s`
- `max_angular_velocity: 1.0 rad/s`
- `max_single_move_distance: 5.0 m`
- Battery thresholds (low: 20%, critical: 10%)

**Safety Guards** ([safety/guard.py](ranger_llm_ui/safety/guard.py)):
- Velocity clamping
- Distance validation
- Battery checks

**Agent Prompts** ([ranger_prompts.py](ranger_llm_ui/ranger_prompts.py)):
- Emergency stop instructions
- Movement confirmation requirements
- Honest status reporting

### 5. LLM Backends

The system supports multiple LLM providers via [agent_interface.py](ranger_llm_ui/agent_interface.py:54):

**OpenAI** (default):
```python
provider="openai", model="gpt-4"
```

**Ollama** (local models):
```python
provider="ollama", model="llama2"
```

**Anthropic** (Claude):
```python
provider="anthropic", model="claude-3-sonnet-20240229"
```

Configuration via environment variables or [default_config.yaml](config/default_config.yaml).

### 6. ROS 2 Integration

The system is both a Python package AND a ROS 2 package:

**ROS 2 Node** ([ui_node.py](ranger_llm_ui/ui_node.py)):
- Publishes to `/cmd_vel` (geometry_msgs/Twist)
- Subscribes to `/odom`, `/battery_state`
- Integrates with ROS 2 launch files

**Simulation Mode:**
Run without ROS 2 using `--simple` flag for testing.

## Running the Project

### Installation

```bash
# Clone with submodules (IMPORTANT!)
git clone --recurse-submodules https://github.com/anh0001/ranger-llm-ui.git
cd ranger-llm-ui

# Initialize submodules if not already done
git submodule update --init

# Install dependencies
pip install -r requirements.txt
pip install -e ros-technician-cli/  # Install ROSA
pip install -e .                     # Install ranger_llm_ui
```

### Environment Variables

Create `.env` file:
```bash
# For OpenAI
OPENAI_API_KEY=your_key_here

# For Ollama (local)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434

# For Anthropic
ANTHROPIC_API_KEY=your_key_here
LLM_PROVIDER=anthropic
```

### Launch Options

**Simple Mode (no ROS 2):**
```bash
python -m ranger_llm_ui.ui_node --simple
```

**With ROS 2:**
```bash
# Build workspace
cd ~/ros2_ws
colcon build --packages-select ranger_llm_ui
source install/setup.bash

# Run node
ros2 run ranger_llm_ui ui_node

# Or use launch file
ros2 launch ranger_llm_ui ranger_llm_ui.launch.py
```

UI available at: `http://localhost:7860`

## Development Guidelines

### Code Organization

- **Tools** - Each tool should be self-contained in `tools/`. Inherit from `BaseTool`.
- **ROS Interface** - Keep ROS-specific code isolated for easy simulation mode.
- **Safety** - Always validate inputs in tools. Never trust agent output directly.
- **Prompts** - Changes to agent behavior should be in prompts, not code.

### Adding New Tools

1. Create tool class inheriting `BaseTool` in `tools/`
2. Define `name`, `description`, `args_schema` (Pydantic model)
3. Implement `_run()` method
4. Register in [all_tools.py](ranger_llm_ui/tools/all_tools.py)
5. Update prompts to describe new capability

Example:
```python
class MyNewTool(BaseTool):
    name = "MyNewTool"
    description = "What this tool does"
    args_schema = MyToolInput  # Pydantic model

    def _run(self, arg1: str) -> str:
        # Implementation
        return "Result"
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_tools.py -v

# Code style
black ranger_llm_ui/
flake8 ranger_llm_ui/
mypy ranger_llm_ui/
```

### Debugging

**Agent Behavior:**
- Check prompts in [ranger_prompts.py](ranger_llm_ui/ranger_prompts.py)
- Enable verbose mode: `RangerAgent(verbose=True)`
- Review logs in `~/.ranger_llm_ui/logs/`

**Tool Issues:**
- Test tools directly: `tool._run(test_input)`
- Check ROS interface initialization
- Verify ROS topics are publishing

**LLM Issues:**
- Test with different providers/models
- Check API keys in `.env`
- Try `simple_mode=True` to bypass LLM

## Important Notes for AI Assistants

### When Modifying This Codebase:

1. **Submodules:** Never edit files in `ros-technician-cli/` or `ros2_numpy/` - these are git submodules.

2. **ROSA Integration:** The `RangerAgent` class WRAPS ROSA, it doesn't reimplement it. Changes to agent core logic should respect ROSA's API.

3. **Safety Critical:** Changes to movement tools, safety guards, or velocity limits require careful review. Always maintain safety constraints.

4. **Prompts vs Code:** Prefer changing agent behavior via prompts rather than code. The agent persona is defined in prompts.

5. **ROS 2 Compatibility:** Maintain ability to run with and without ROS 2 (simulation mode). Don't hardcode ROS dependencies.

6. **Tool Independence:** Tools should work independently of the agent for testing. Each tool should validate its own inputs.

### Common Tasks:

**Change Robot Behavior:**
→ Edit [ranger_prompts.py](ranger_llm_ui/ranger_prompts.py)

**Add New Movement Command:**
→ Add tool in [movement_tools.py](ranger_llm_ui/tools/movement_tools.py), register in [all_tools.py](ranger_llm_ui/tools/all_tools.py)

**Adjust Safety Limits:**
→ Edit [default_config.yaml](config/default_config.yaml) and [safety/guard.py](ranger_llm_ui/safety/guard.py)

**Change LLM Provider:**
→ Set environment variable `LLM_PROVIDER=anthropic` or edit config

**Fix UI Issues:**
→ Check [ui_node.py](ranger_llm_ui/ui_node.py) Gradio interface

### Testing Without Hardware:

Use `--simple` mode for testing without ROS 2:
```bash
python -m ranger_llm_ui.ui_node --simple
```

This uses `SimpleAgent` ([agent_interface.py](ranger_llm_ui/agent_interface.py:300)) which parses basic commands without an LLM.

## Dependencies

**Core:**
- Python 3.10+
- ROS 2 Humble (optional for simulation mode)
- LangChain (~0.3.23)
- Gradio (>=4.0.0)

**LLM Providers:**
- `langchain-openai` (~0.3.14) - OpenAI integration
- `langchain-ollama` (~0.3.2) - Local models
- `langchain-anthropic` - Claude (optional)

**Submodules:**
- `ros-technician-cli` - ROSA framework
- `ros2_numpy` - NumPy conversions for ROS messages

## References

- [ROSA (NASA JPL)](https://github.com/nasa-jpl/rosa) - Base agent framework
- [LangChain Documentation](https://python.langchain.com/) - Agent and tools
- [Gradio Guide](https://www.gradio.app/guides/agents-and-tool-usage) - UI patterns
- [ROS 2 Humble Docs](https://docs.ros.org/en/humble/) - ROS concepts

## License

MIT License - See [LICENSE](LICENSE) file

## Project Maintainer

Anh Nguyen - anh0001@example.com

---

**Last Updated:** 2026-01-18
**Codebase Version:** 0.1.0
