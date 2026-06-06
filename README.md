# Ranger LLM UI

LLM-driven natural language operator interface for the Ranger garden robot.

## Features

- **Natural Language Control**: Control the robot using conversational commands like "move forward 1 meter" or "turn left 90 degrees"
- **Safety First**: Built-in safety guards with velocity limits, distance validation, and emergency stop
- **Multiple LLM Backends**: OpenAI, Ollama (local), Anthropic API, and **Claude Pro/Max subscription via OAuth** (no API key needed)
- **Voice I/O (optional)**: Hands-free speech-to-text mic input (faster-whisper) and spoken replies (Piper / Kokoro / espeak-ng) — fully local, no cloud key
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
# Pick the asset matching your OS/arch from the releases page (use a current tag):
# https://github.com/router-for-me/CLIProxyAPI/releases
VER=7.1.31
ARCH=amd64           # x86_64 desktop/server; use 'arm64' on Jetson/aarch64
curl -sL -o p.tar.gz "https://github.com/router-for-me/CLIProxyAPI/releases/download/v${VER}/CLIProxyAPI_${VER}_linux_${ARCH}.tar.gz"
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
# IMPORTANT: pin to the 0.3.x line. The unpinned latest (1.4.x) drags
# langchain-core up to 1.x and breaks ROSA's `langchain-core~=0.3.52` pin.
pip install --user "langchain-anthropic~=0.3.13"
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

## Voice (Speech-to-Text & Text-to-Speech)

Optional, fully local/offline voice. Tap the 🎤 mic button next to the command
box, speak, tap stop — your speech is transcribed and sent automatically, and
the robot's reply is spoken back. Toggle spoken replies with the **🔊 Speak
replies** checkbox.

### Speech-to-Text (STT) — works out of the box

STT uses **faster-whisper** and is installed by `requirements.txt`
(`faster-whisper`, `av`, `soundfile`). No extra step. The first transcription
lazily loads the model. Decoding is tuned for short, fixed-vocabulary commands:
beam search (`beam_size=5`), deterministic (`temperature=0`), and biased toward
the robot's command words via a built-in `initial_prompt` + `hotwords`.

```bash
export WHISPER_MODEL=small.en   # default; use medium.en for best accuracy on a GPU box
export WHISPER_BEAM_SIZE=5      # 1 = greedy (faster, less accurate)
export WHISPER_HOTWORDS=""      # override/disable the built-in command-word boost
export WHISPER_INITIAL_PROMPT="" # override/disable the built-in domain prompt
export WHISPER_VAD=1            # optional voice-activity trimming (pulls onnxruntime)
```

#### LLM second-pass correction (uses Claude)

After Whisper, an optional second pass sends **low-confidence** transcripts to
the app's LLM (the same provider as the chat agent) to fix obvious ASR errors
against the command vocabulary — e.g. `turn write ninety degrees` →
`turn right ninety degrees`, `what's the batter level` → `battery status`.
It is gated so it stays cheap and safe:

- **Stop commands bypass the LLM** entirely (never rewritten).
- **High-confidence transcripts skip it** (no added latency).
- It **never invents** a missing distance/angle/destination, and falls back to
  the raw transcript on any error.

```bash
export VOICE_LLM_CORRECTION=1            # on by default; 0 to disable
export VOICE_CORRECTION_MODEL=haiku-4.5  # default: the chat model; haiku = faster/cheaper
```

### Text-to-Speech (TTS) — install a backend, or replies are silent

> **Heads-up:** no TTS backend ships enabled by default (kokoro/piper/espeak are
> commented out in `requirements.txt`). Without one, replies have **no sound** —
> the log shows `TTS skipped: ... not installed`. Install **one** of:

```bash
export PYTHONUSERBASE="$HOME/.local/ranger_llm_ui_py310"

# Piper — natural, lightweight, recommended on x86/desktop (needs healthy onnxruntime).
# First spoken reply auto-downloads the voice (~60 MB) to ~/.ranger_llm_ui/voices.
pip install --user piper-tts

# Kokoro — most natural (PyTorch, avoids onnxruntime). Best on Jetson. Heavier.
pip install --user kokoro==0.9.4 soundfile "misaki[en]"

# espeak-ng — robotic but rock-solid fallback, zero Python deps.
sudo apt install espeak-ng
```

Backend auto-selection is **Kokoro → Piper → espeak-ng**. Force one with
`TTS_BACKEND=piper|kokoro|espeak`. Voices: `PIPER_VOICE` (default
`en_US-lessac-medium`), `KOKORO_VOICE` (default `af_heart`),
`ESPEAK_VOICE` (default `en-us`).

### Browser microphone requires a secure context (HTTPS or localhost)

Browsers only grant mic access (`getUserMedia`) on a **secure origin**. Opening
the UI over plain HTTP at a LAN/Tailscale IP (e.g. `http://10.0.0.5:7860`) will
**never prompt for the mic**. Use one of:

- **Localhost / SSH tunnel** (simplest): `ssh -L 7860:127.0.0.1:7860 user@host`
  then open `http://localhost:7860` — `localhost` is a secure context.
- **HTTPS via Tailscale Serve** (works from any device incl. iPhone/Safari):
  ```bash
  # one-time: enable "HTTPS Certificates" in the Tailscale admin console (DNS page)
  tailscale serve --bg --https=443 http://127.0.0.1:7860
  ```
  then open `https://<magicdns-name>.ts.net/`. Set `GRADIO_SERVER_NAME=127.0.0.1`
  so the UI is reachable only through the HTTPS proxy.

> **iOS Safari autoplay:** spoken replies won't auto-play until you've tapped the
> page once. Tap anywhere, then replies play automatically.

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
| `GRADIO_SERVER_NAME` | Bind address (`127.0.0.1` to expose only via an HTTPS proxy) | 0.0.0.0 |
| `WHISPER_MODEL` | STT model (base.en, small.en, medium.en, ...) | small.en |
| `WHISPER_BEAM_SIZE` | STT beam search width (1 = greedy) | 5 |
| `VOICE_LLM_CORRECTION` | LLM second-pass transcript correction | 1 (on) |
| `VOICE_CORRECTION_MODEL` | Model for the correction call | chat model |
| `TTS_BACKEND` | TTS engine: auto, kokoro, piper, espeak | auto |
| `PIPER_VOICE` | Piper voice name | en_US-lessac-medium |
| `KOKORO_VOICE` | Kokoro voice name | af_heart |
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
- **NavigateToPose**: Navigate to a target pose
- **StopRobot**: Emergency stop

### Status Tools
- **BatteryStatus**: Get battery level and charging status
- **SystemHealth**: Check overall system health
- **GetOdometry**: Get current position and velocity
- **GetCameraImage**: Capture a camera image (token-optimized; see CLAUDE.md)
- **ListNodes**: List active ROS 2 nodes
- **ListTopics**: List active ROS 2 topics

### Manipulation Tools (arm)
Executed by the companion [MobileManipulationCore](https://github.com/anh0001/MobileManipulationCore)
skill server over a single ROS 2 action (`manipulation_msgs/action/ExecuteSkill`
on `/execute_skill`):
- **Pick**: Grasp an object by open-vocabulary name (e.g. "pick up the bread")
- **Place**: Release the held object onto/into a named target (run after Pick)
- **PickAndPlace**: Pick an object and place it at a destination in one call
- **HomeArm**: Move the arm to a named pose (`ready` or `rest`)
- **Handover**: Hand the held object to a person, then open the gripper

> Requires the MobileManipulationCore stack: its `manipulation_msgs` overlay must
> be **sourced before the UI starts** (the action client is wired up at import
> time), and its `skill_server` must be running (`ros2 launch
> manipulation_bringup core_launch.py`, on top of the `ranger-garden-assistant`
> base bringup).
>
> **Sourcing the overlay:** if you launch through `scripts/dev.sh` (the VS Code
> tasks do), the sibling `../MobileManipulationCore/install` is detected and
> sourced automatically — no setup needed when the standard sibling layout is in
> place. Point elsewhere with `MMC_INSTALL=/path/to/MobileManipulationCore/install`,
> or set `MMC_INSTALL=` (empty) to skip the overlay. Launching by hand instead of
> via `dev.sh`? Source it yourself first:
> ```bash
> source /opt/ros/humble/setup.bash
> source ../MobileManipulationCore/install/setup.bash   # provides manipulation_msgs
> ```
> If `manipulation_msgs` is missing, Pick/Place report "manipulation system
> unavailable"; if the overlay is sourced but `skill_server` isn't running, they
> report "`/execute_skill` action server not available".
>
> The tools are simulated in `--simple` mode and fail gracefully (with a hint) if
> the stack is absent. Disable with `ENABLE_MANIPULATION_TOOLS=false`; remap the
> action with `EXECUTE_SKILL_ACTION`.

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
- [x] Claude Pro/Max subscription backend (claude_proxy)
- [x] Voice interface (local STT mic input + spoken TTS replies)
- [x] Camera/sensor visualization (Status tab camera view)
- [ ] Multi-step task automation
- [ ] Confirmation dialogs for dangerous actions
- [ ] Command history panel
- [ ] Ollama integration testing
- [ ] Enhanced telemetry dashboard

### Future
- [x] Arm manipulation skills (Pick/Place/Handover via MobileManipulationCore)
- [ ] Nav2 navigation integration
- [ ] Learning from corrections

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file