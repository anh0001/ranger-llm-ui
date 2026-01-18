#!/bin/bash
# Ranger LLM UI Environment Setup Script (system Python + isolated PYTHONUSERBASE)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}Ranger LLM UI Environment Setup${NC}"
echo -e "${GREEN}====================================${NC}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Require python3 and ROS Humble
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}Error: python3 not found. Install system Python 3.10 (default on Ubuntu 22.04).${NC}"
    exit 1
fi
if [ ! -f "/opt/ros/humble/setup.bash" ]; then
    echo -e "${RED}Error: ROS 2 Humble not found at /opt/ros/humble/setup.bash${NC}"
    echo "Install ROS 2 Humble before running this script."
    exit 1
fi

PYTHON_BIN="$(command -v python3)"
DEFAULT_USERBASE="$HOME/.local/ranger_llm_ui_py310"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$DEFAULT_USERBASE}"
mkdir -p "$PYTHONUSERBASE"

echo -e "${YELLOW}Using PYTHONUSERBASE=${PYTHONUSERBASE}${NC}"
echo -e "${YELLOW}Add 'export PYTHONUSERBASE=${PYTHONUSERBASE}' to your shell profile for future terminals.${NC}"

pip_user() {
    "$PYTHON_BIN" -m pip install --user "$@"
}

echo -e "\n${YELLOW}Step 1: Upgrading pip in the isolated user base${NC}"
"$PYTHON_BIN" -m pip install --upgrade --user pip

echo -e "\n${YELLOW}Step 2: Installing Python dependencies from requirements.txt${NC}"
pip_user -r "$SCRIPT_DIR/requirements.txt"

echo -e "\n${YELLOW}Step 3: Installing ros-technician-cli (ROSA) as editable package${NC}"
if [ -d "$SCRIPT_DIR/ros-technician-cli" ]; then
    pip_user -e "$SCRIPT_DIR/ros-technician-cli"
else
    echo -e "${RED}Warning: ros-technician-cli submodule not found. Run 'git submodule update --init' first.${NC}"
fi

echo -e "\n${YELLOW}Step 4: Installing ranger_llm_ui as editable package${NC}"
pip_user -e "$SCRIPT_DIR"

echo -e "\n${YELLOW}Step 5: Installing ROS 2 dependencies via rosdep${NC}"
set +u  # ROS setup scripts reference unset vars such as AMENT_TRACE_SETUP_FILES
source /opt/ros/humble/setup.bash
set -u
rosdep install --from-paths "$SCRIPT_DIR" "$SCRIPT_DIR/ros2_numpy" --ignore-src -y || true

echo -e "\n${YELLOW}Step 6: Cleaning previous colcon build artifacts (safe if absent)${NC}"
rm -rf "$SCRIPT_DIR/build" "$SCRIPT_DIR/install" "$SCRIPT_DIR/log"

echo -e "\n${YELLOW}Step 7: Building ROS 2 packages${NC}"
cd "$SCRIPT_DIR"
colcon build --packages-select ros2_numpy ranger_llm_ui

echo -e "\n${YELLOW}Step 8: Verifying imports${NC}"
set +u
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"
set -u
"$PYTHON_BIN" - <<'VERIFY'
import gradio
import langchain
import langchain_openai
import langchain_ollama
print("Gradio:", gradio.__version__)
print("LangChain:", langchain.__version__)
print("Core dependencies import OK")
VERIFY

echo -e "\n${GREEN}====================================${NC}"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo -e "${GREEN}====================================${NC}"
cat <<EOF

Next steps for each terminal:
1. export PYTHONUSERBASE="$PYTHONUSERBASE"    # add to ~/.bashrc for convenience
2. source /opt/ros/humble/setup.bash
3. source $SCRIPT_DIR/install/setup.bash

Optional helper alias:
    echo 'export PYTHONUSERBASE="$PYTHONUSERBASE"' >> ~/.bashrc
    echo 'alias ranger_setup="export PYTHONUSERBASE=$PYTHONUSERBASE && source /opt/ros/humble/setup.bash && source $SCRIPT_DIR/install/setup.bash"' >> ~/.bashrc

To run the UI:
    python -m ranger_llm_ui.ui_node --simple    # Without ROS 2
    ros2 run ranger_llm_ui ui_node              # With ROS 2

EOF
