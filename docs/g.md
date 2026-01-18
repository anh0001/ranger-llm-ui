# Ranger LLM UI - AI Agent Instructions

## Project Overview

**Ranger LLM UI** is a natural language control interface for the Ranger garden robot. It combines:
- **NASA JPL's ROSA** (from `ros-technician-cli/` git submodule) - provides ROS2 introspection tools and LangChain agent framework
- **Ranger-specific tools** - robot movement and status control via ROS2
- **Gradio web UI** - streaming chat interface for operator interaction
- **Safety guards** - velocity limits, distance validation, and emergency stop

**Key Architecture Principle**: ROSA is the base agent (wraps LangChain), extended with Ranger tools. Never bypass ROSA - always work through its abstraction.

## Critical File Structure

```
ranger_llm_ui/
├── agent_interface.py       # RangerAgent wraps ROSA, adds Ranger tools
├── ui_node.py              # Entry point: ROS2 node + Gradio server
├── ranger_prompts.py       # Robot persona using ROSA's RobotSystemPrompts
├── tools/
│   ├── movement_tools.py   # MoveForward/Backward/Turn/Stop (LangChain tools)
│   ├── status_tools.py     # Battery/Health/Odometry queries
│   └── all_tools.py        # Tool registry and initialization
├── safety/
│   └── guard.py            # SafetyGuard validates all movements
└── schemas/
    └── commands.py         # Dataclasses for movement/turn/stop commands

ros-technician-cli/src/rosa/  # Git submodule - DO NOT MODIFY directly
├── rosa.py                 # ROSA class (agent executor, streaming)
├── prompts.py              # RobotSystemPrompts base class
└── tools/                  # ROS2 introspection tools (topics, nodes, logs)
```

## Development Workflows

### Building & Running

**Without ROS2** (simulation mode for UI testing):
```bash
python -m ranger_llm_ui.ui_node --simple  # Uses SimpleAgent (no LLM)
python -m ranger_llm_ui.ui_node           # Uses RangerAgent with LLM
```

**With ROS2** (requires `colcon` workspace):
```bash
cd ~/ros2_ws
colcon build --packages-select ranger_llm_ui
source install/setup.bash
ros2 launch ranger_llm_ui ranger_llm_ui.launch.py
# UI at http://localhost:7860
```

**Testing**:
```bash
pytest tests/                           # All tests (work in simulation)
pytest tests/test_tools.py -k movement  # Specific test subset
```

### LLM Provider Configuration

Set via environment variables (`.env` file supported):
```bash
# OpenAI (default)
export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4  # Optional, defaults to gpt-4

# Anthropic (Claude)
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-3-sonnet-20240229  # Default model

# Ollama (local)
export LLM_PROVIDER=ollama
export LLM_MODEL=llama2
export OLLAMA_BASE_URL=http://localhost:11434
```

See `create_llm()` in [agent_interface.py](agent_interface.py#L56-L126) for implementation.

## Code Conventions

### 1. Tool Creation Pattern

All tools are **LangChain `BaseTool`** subclasses with Pydantic input schemas:

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class MoveInput(BaseModel):
    distance_m: float = Field(description="Distance in meters (positive only)")

class MoveForwardTool(BaseTool):
    name = "MoveForward"
    description = "Move robot forward. Use for explicit 'move forward' commands."
    args_schema: Type[BaseModel] = MoveInput
    
    def _run(self, distance_m: float) -> str:
        # Safety validation happens here
        guard = get_safety_guard()
        validated = guard.validate_distance(distance_m)
        # Execute via ROSInterface singleton
        interface = get_ros_interface()
        return interface.execute_move(...)
```

**Critical**: Tools MUST validate inputs via `SafetyGuard` before execution. See [movement_tools.py](ranger_llm_ui/tools/movement_tools.py#L150-L200).

### 2. ROS2 Communication Pattern

Use **singleton interfaces** to manage ROS2 lifecycle:
- `ROSInterface` (movement) - initialized once with ROS2 node
- `StatusInterface` (sensors) - initialized once with ROS2 node

```python
# In ui_node.py startup
from ranger_llm_ui.tools.movement_tools import initialize_ros_interface
initialize_ros_interface(self._node)  # Pass rclpy node

# In tools
ros = get_ros_interface()  # Retrieves singleton
ros.publish_velocity(linear_x=0.2, angular_z=0.0)
```

**Simulation mode**: When `node=None`, interfaces run in simulation (print actions, no ROS).

### 3. Safety-First Movement

Every movement tool follows this sequence:
1. **Validate** via `SafetyGuard.validate_distance()` or `validate_velocity()`
2. **Check emergency stop** via `SafetyGuard.state.emergency_stop_active`
3. **Execute** via `ROSInterface` with confirmed safe parameters
4. **Log** via command logger

See [safety/guard.py](ranger_llm_ui/safety/guard.py#L45-L80) for limits:
- Max linear velocity: 0.5 m/s
- Max angular velocity: 1.0 rad/s  
- Max single move: 5.0 m
- Confirmation threshold: 2.0 m

### 4. Prompt Engineering with ROSA

Ranger's persona is defined in [ranger_prompts.py](ranger_llm_ui/ranger_prompts.py) using `RobotSystemPrompts`:

```python
from rosa import RobotSystemPrompts

prompts = RobotSystemPrompts(
    embodiment_and_persona="You are Ranger, a garden robot...",
    critical_instructions="SAFETY-CRITICAL: Always execute StopRobot on 'stop'...",
    constraints_and_guardrails="YOU MUST NOT: Execute arbitrary code...",
    about_your_environment="Garden terrain, /cmd_vel topic..."
)
```

**When modifying prompts**:
- Keep `critical_instructions` focused on safety
- Reference actual tool names (e.g., "StopRobot tool")
- Describe ROS2 topics/services the robot uses

### 5. Streaming Response Pattern

RangerAgent uses ROSA's async streaming for Gradio UI:

```python
async for event in agent.astream(user_input):
    if event["type"] == "token":
        yield event["content"]  # Partial LLM response
    elif event["type"] == "tool_start":
        yield f"🔧 {event['tool']}..."  # Tool execution feedback
    elif event["type"] == "final":
        yield event["output"]  # Complete response
```

See [agent_interface.py](ranger_llm_ui/agent_interface.py#L248-L280) and [ui_node.py](ranger_llm_ui/ui_node.py#L200-L250).

## Integration Points

### ROSA Submodule (ros-technician-cli)

- **Location**: `ros-technician-cli/` (git submodule at branch `ranger-garden-assistant`)
- **Purpose**: Provides `ROSA` agent class and ROS2 introspection tools
- **DO NOT**: Modify submodule files directly - make changes upstream at nasa-jpl/rosa
- **Integration**: Dynamically added to `sys.path` in `agent_interface.py` and `ranger_prompts.py`

```python
# Pattern used in multiple files
_submodule_src = os.path.join(os.path.dirname(__file__), '..', 'ros-technician-cli', 'src')
if os.path.exists(_submodule_src):
    sys.path.insert(0, os.path.abspath(_submodule_src))
from rosa import ROSA, RobotSystemPrompts
```

### ROS2 Topics & Messages

**Movement** (`/cmd_vel` - `geometry_msgs/Twist`):
- `linear.x`: forward/backward velocity (m/s)
- `angular.z`: rotation velocity (rad/s)

**Odometry** (`/odom` - `nav_msgs/Odometry`):
- Position tracking for distance-based movements

**Battery** (`/battery_state` - `sensor_msgs/BatteryState`):
- Percentage, voltage for status queries

### External Dependencies

- **LangChain 0.1.x**: Agent framework, tool binding
- **Gradio 4.x**: Web UI with streaming chat
- **rclpy**: ROS2 Python client (optional in sim mode)

## Common Pitfalls

1. **Don't bypass ROSA**: Never create raw LangChain agents - always use `ROSA` class from submodule
2. **Initialize interfaces**: Tools fail silently if `initialize_ros_interface()` not called before agent creation
3. **Simulation vs ROS**: Check `ROS_AVAILABLE` flag before assuming ROS2 functionality
4. **Safety validation**: Movements without `SafetyGuard` checks will be rejected by operator
5. **Async context**: Gradio callbacks run in asyncio loop - use `agent.astream()` not `agent.invoke()`

## Testing Notes

- **All tests** run in simulation mode (no ROS2 required)
- **Movement tools** mock `ROSInterface` to verify safety checks
- **Status tools** return simulated battery/health data
- **SimpleAgent** (no LLM) available for tool infrastructure testing: `create_agent(simple_mode=True)`

## Key Files to Reference

- [agent_interface.py](ranger_llm_ui/agent_interface.py) - Agent creation, LLM provider switching
- [tools/movement_tools.py](ranger_llm_ui/tools/movement_tools.py) - Movement tool implementation pattern
- [safety/guard.py](ranger_llm_ui/safety/guard.py) - Safety limits and validation logic
- [ui_node.py](ranger_llm_ui/ui_node.py) - Gradio UI setup, streaming integration
- [ros-technician-cli/src/rosa/rosa.py](ros-technician-cli/src/rosa/rosa.py) - ROSA base agent (read-only reference)
