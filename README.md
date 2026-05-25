# Ranger LLM UI

LLM-driven natural language operator interface for the Ranger garden robot.

## Features

- **Natural Language Control**: Control the robot using conversational commands like "move forward 1 meter" or "turn left 90 degrees"
- **Safety First**: Built-in safety guards with velocity limits, distance validation, and emergency stop
- **Multiple LLM Backends**: OpenAI, Ollama (local), Anthropic API, and **Claude Pro/Max subscription via OAuth** (no API key needed)
- **ROS 2 Integration**: Native ROS 2 Humble support via rclpy
- **Manual Override**: Emergency stop button and manual teleop controls always available

## Architecture

```
User (Operator)
     │ Types NL command
     ▼
[Chat UI] ──(LLM prompt)──> [LangChain Agent]
     │                                    │
     │                                    ▼
     │                            [Ranger Tools]
     │                                    │
     │                                    ▼
     │                            [ROS 2 Interface]
     │                                    │
     ◄────────(Feedback/Status)───────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- ROS 2 Humble (optional - can run in simulation mode)
- OpenAI API key (or Ollama for local models)

### Installation

**Automated Setup (Recommended):**

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/anh0001/ranger-llm-ui.git
cd ranger-llm-ui

# Run the automated setup script
./scripts/setup_env.sh
```

The setup script will:
- ✓ Create an isolated Python user environment
- ✓ Install all Python dependencies
- ✓ Install ROSA (ros-technician-cli) submodule
- ✓ Install ROS 2 dependencies via rosdep
- ✓ Build ROS 2 packages (ros2_numpy, ranger_llm_ui)
- ✓ Verify all imports

After installation, source the environment in each terminal:
```bash
export PYTHONUSERBASE="$HOME/.local/ranger_llm_ui_py310"
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Or add a convenient alias to your `~/.bashrc`:
```bash
echo 'alias ranger_setup="export PYTHONUSERBASE=$HOME/.local/ranger_llm_ui_py310 && source /opt/ros/humble/setup.bash && source $(pwd)/install/setup.bash"' >> ~/.bashrc
```

**Configure LLM Provider:**

Create a `.env` file for API keys:
```bash
# For OpenAI (default)
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "LLM_MODEL=gpt-4o-mini" >> .env  # cheaper default for testing

# Or for Ollama (local models):
echo "LLM_PROVIDER=ollama" >> .env
echo "OLLAMA_BASE_URL=http://localhost:11434" >> .env

# Or for Anthropic (Claude, pay-per-token API key):
echo "LLM_PROVIDER=anthropic" >> .env
echo "ANTHROPIC_API_KEY=your_api_key_here" >> .env

# Or for Claude Pro/Max subscription via CLIProxyAPI (no API key, OAuth):
# See "Claude Pro/Max Subscription Setup" section below.
echo "LLM_PROVIDER=claude_proxy" >> .env
echo "LLM_MODEL=sonnet-4.6" >> .env
echo "CLAUDE_PROXY_BASE_URL=http://127.0.0.1:8317" >> .env
echo "CLAUDE_PROXY_API_KEY=ranger-local-key" >> .env
```

### Claude Pro/Max Subscription Setup (claude_proxy)

Use your existing Claude Pro/Max subscription as the LLM backend. No
separate Anthropic API credits. Tool calling works end-to-end via LangChain
`bind_tools()`. Routes through [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI)
which wraps Claude Code OAuth as an Anthropic-compatible HTTP endpoint.

**1. Download CLIProxyAPI** (Go binary, prebuilt for linux/macOS):
```bash
mkdir -p ~/tools/cliproxyapi && cd ~/tools/cliproxyapi
# Pick the asset matching your OS/arch from the releases page:
# https://github.com/router-for-me/CLIProxyAPI/releases
curl -sL -o p.tar.gz https://github.com/router-for-me/CLIProxyAPI/releases/download/v7.1.20/CLIProxyAPI_7.1.20_linux_aarch64.tar.gz
tar xzf p.tar.gz && rm p.tar.gz
cp config.example.yaml config.yaml
# Edit config.yaml to set api-keys to a single value, e.g. "ranger-local-key"
```

**2. Authenticate via Claude OAuth (one-time):**
```bash
./cli-proxy-api -claude-login -no-browser -config config.yaml
# Follow printed instructions:
#  - On your local machine, SSH-tunnel localhost:54545
#      ssh -L 54545:127.0.0.1:54545 <user>@<server>
#  - Open the printed claude.ai URL in your local browser
#  - Log in with your Pro/Max account, authorize
# Credentials saved to ~/.cli-proxy-api/claude-<email>.json
```

**3. Run the proxy:**
```bash
nohup ./cli-proxy-api -config config.yaml > /tmp/cliproxy.log 2>&1 &
# Listens on http://127.0.0.1:8317 by default.
```

**4. Install the LangChain Anthropic adapter** (one-time, in your ranger env):
```bash
export PYTHONUSERBASE="$HOME/.local/ranger_llm_ui_py310"
pip install langchain-anthropic
```

**5. Run ranger-llm-ui pointing at the proxy:**
```bash
python -m ranger_llm_ui.ui_node --provider claude_proxy --model sonnet-4.6
# Or set LLM_PROVIDER=claude_proxy in .env and launch normally.
```

**Available model aliases for claude_proxy:**
`sonnet-4.6` (default), `sonnet-4.5`, `sonnet-4`, `opus`, `haiku-4.5`

**Caveats:**
- Counts against your Claude Pro/Max usage limit (same pool as Claude Code CLI).
- Starting **2026-06-15**, Agent SDK / `claude -p` subscription usage draws from a separate monthly credit (Pro: $20/mo, Max 5x: $100/mo, Max 20x: $200/mo). Overage stops requests.
- ToS gray area: unofficial wrapper. Can break if Anthropic updates Claude Code internals.
- For production robot deployments with strict uptime needs, prefer `provider=anthropic` with a real API key.

### Cost Control (OpenAI)

This UI uses a tool-calling agent (ROSA), which can make multiple model calls per message and re-send chat history + tool schemas each time. If you see unexpectedly high usage, set:
- `LLM_MODEL=gpt-4o-mini` (or another cheaper model)
- `ROSA_MAX_ITERATIONS=15` (limits agent looping)
- `ROSA_MAX_HISTORY_MESSAGES=20` (limits history growth)
- `LLM_MAX_TOKENS=512` (caps long answers)
- `SHOW_LLM_USAGE=true` (shows token counts in the chat)

### Running

**Without ROS 2 (Simulation Mode):**
```bash
python -m ranger_llm_ui.ui_node --simple
```

**With ROS 2:**
```bash
# Build with colcon
cd ~/ros2_ws
colcon build --packages-select ranger_llm_ui
source install/setup.bash

# Run the UI node
ros2 run ranger_llm_ui ui_node

# Or use the launch file
ros2 launch ranger_llm_ui ranger_llm_ui.launch.py
```

The UI will be available at `http://localhost:7860`

## Usage

### Example Commands

| Command | Action |
|---------|--------|
| "Move forward 1 meter" | Moves the robot forward |
| "Turn left 90 degrees" | Rotates counterclockwise |
| "What's your battery level?" | Reports battery status |
| "Check system health" | Runs diagnostics |
| "Stop" | Emergency stop |

### Manual Controls

The UI provides manual teleop buttons for direct control:
- Arrow buttons for movement
- Emergency stop button (always visible)
- Battery status display

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM backend (openai, ollama, anthropic, claude_code, claude_proxy) | openai |
| `LLM_MODEL` | Model name (claude_proxy: sonnet-4.6, opus, haiku-4.5, ...) | provider default |
| `CLAUDE_PROXY_BASE_URL` | CLIProxyAPI URL (claude_proxy only) | http://127.0.0.1:8317 |
| `CLAUDE_PROXY_API_KEY` | CLIProxyAPI key from its config.yaml (claude_proxy only) | ranger-local-key |
| `CLAUDE_CODE_OAUTH_TOKEN` | OAuth token for direct Claude Code SDK (claude_code only) | unset |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OLLAMA_BASE_URL` | Ollama server URL | http://localhost:11434 |
| `GRADIO_PORT` | Web UI port | 7860 |
| `LLM_MAX_TOKENS` | Max completion tokens (OpenAI) | unset |
| `ROSA_MAX_ITERATIONS` | Max agent iterations per message | 15 |
| `ROSA_MAX_HISTORY_MESSAGES` | Max messages kept in agent history | 20 |
| `SHOW_LLM_USAGE` | Show token usage in chat | false |

### Configuration File

See `config/default_config.yaml` for all available options including safety limits.

## Project Structure

```
ranger-llm-ui/
├── ranger_llm_ui/           # Main package
│   ├── ui_node.py           # Entry point
│   ├── agent_interface.py   # LangChain agent
│   ├── tools/               # Robot tools
│   │   ├── movement_tools.py
│   │   ├── status_tools.py
│   │   └── all_tools.py
│   ├── schemas/             # Command dataclasses
│   ├── safety/              # Safety guards
│   └── utils/               # Logging utilities
├── launch/                  # ROS 2 launch files
├── config/                  # Configuration files
├── tests/                   # Test suite
├── package.xml              # ROS 2 package manifest
└── pyproject.toml           # Python project config
```

## Available Tools

### Movement Tools
- **MoveForward**: Move forward by specified distance (meters)
- **MoveBackward**: Move backward by specified distance
- **TurnAngle**: Rotate in place (positive = right, negative = left)
- **StopRobot**: Emergency stop

### Status Tools
- **BatteryStatus**: Get battery level and charging status
- **SystemHealth**: Check overall system health
- **GetOdometry**: Get current position and velocity
- **ListNodes**: List active ROS 2 nodes
- **ListTopics**: List active ROS 2 topics

## Safety Features

1. **Velocity Limits**: All velocities are clamped to safe maximums
2. **Distance Validation**: Large movements require confirmation
3. **Emergency Stop**: Always available via UI button
4. **Battery Monitoring**: Warns on low battery, stops on critical
5. **Tool Constraints**: Agent can only use predefined safe tools

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Style
```bash
black ranger_llm_ui/
flake8 ranger_llm_ui/
mypy ranger_llm_ui/
```

## Roadmap

### MVP (Current)
- [x] Natural language commanding
- [x] Chat interface
- [x] Core movement and status tools
- [x] Safety mechanisms
- [x] OpenAI backend

### Near-Term
- [ ] Multi-step task automation
- [ ] Confirmation dialogs for dangerous actions
- [ ] Command history panel
- [ ] Ollama integration testing
- [ ] Enhanced telemetry dashboard

### Future
- [ ] Nav2 navigation integration
- [ ] Voice interface
- [ ] Camera/sensor visualization
- [ ] Learning from corrections

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file