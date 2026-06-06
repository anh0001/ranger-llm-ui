#!/usr/bin/env bash
# Ranger LLM UI — developer task dispatcher.
#
# One place for all the environment plumbing (PYTHONUSERBASE, PYTHONPATH,
# CUDA libs, .env, ROS sourcing) so .vscode/tasks.json — and your terminal —
# stay thin. Run any command below from the repo root or anywhere:
#
#   scripts/dev.sh <command> [extra args...]
#
# Commands:
#   build            Build ROS 2 packages (ranger_llm_msgs + ranger_llm_ui)
#   build-msgs       Build only the ranger_llm_msgs action interfaces
#   clean            Remove build/ install/ log/ artifacts
#   run [args]       Run the Gradio UI via bare `python -m` (not ros2 launch).
#                    Sources the ROS underlay + this workspace + the MMC overlay
#                    first (guarded; no-op without ROS), so real-robot tools
#                    (movement + manipulation) work. Pass --simple to force
#                    pure simulation. Extra args go straight to ui_node, e.g.
#                                                     run --simple
#                                                     run --provider openai --model gpt-4o-mini
#   run-ros [args]   ros2 launch the UI WITH ROS 2 (build first: `dev.sh build`)
#   proxy            Start CLIProxyAPI (Claude Pro/Max OAuth -> Anthropic HTTP)
#   proxy-login      One-time CLIProxyAPI OAuth login
#   test [args]      Run pytest (args pass through, e.g. test -k movement)
#   shell            Open a subshell with the full env set up
#
# Environment overrides (all optional):
#   PYTHONUSERBASE   isolated install dir (default: $HOME/.local/ranger_llm_ui_py310)
#   ROS_DISTRO       ROS 2 distro to source (default: humble)
#   CLIPROXY_DIR     CLIProxyAPI checkout (default: $HOME/tools/cliproxyapi)
#   MMC_INSTALL      MobileManipulationCore install/ to overlay so manipulation
#                    skills (/execute_skill) resolve, e.g.
#                    ../MobileManipulationCore/install
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local/ranger_llm_ui_py310}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
CLIPROXY_DIR="${CLIPROXY_DIR:-$HOME/tools/cliproxyapi}"

# Default the MobileManipulationCore overlay to the documented sibling checkout
# (../MobileManipulationCore) so manipulation skills (/execute_skill,
# manipulation_msgs) resolve with zero config. Override MMC_INSTALL to point
# elsewhere, or set it empty to skip the overlay entirely.
if [[ -z "${MMC_INSTALL+x}" && -f "$REPO_ROOT/../MobileManipulationCore/install/setup.bash" ]]; then
  MMC_INSTALL="$(cd "$REPO_ROOT/../MobileManipulationCore/install" && pwd)"
fi

_site="$PYTHONUSERBASE/lib/python3.10/site-packages"
export PYTHONPATH="$_site${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$_site/nvidia/cu12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WHISPER_MODEL="${WHISPER_MODEL:-base.en}"   # voice STT default

# Load secrets/config from .env if present (OPENAI_API_KEY, LLM_PROVIDER, ...).
load_env() {
  if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
}

# Source the ROS underlay + this workspace overlay (and, if set, a sibling
# MobileManipulationCore overlay). ROS setup scripts reference unbound vars,
# so relax `set -u` while sourcing.
source_ros() {
  set +u
  [[ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]] && source "/opt/ros/$ROS_DISTRO/setup.bash"
  [[ -f "install/setup.bash" ]] && source "install/setup.bash"
  if [[ -n "${MMC_INSTALL:-}" && -f "${MMC_INSTALL}/setup.bash" ]]; then
    source "${MMC_INSTALL}/setup.bash"
  fi
  set -u
}

require_cliproxy() {
  if [[ ! -x "$CLIPROXY_DIR/cli-proxy-api" ]]; then
    echo "CLIProxyAPI binary not found at: $CLIPROXY_DIR/cli-proxy-api" >&2
    echo "Set CLIPROXY_DIR to your CLIProxyAPI checkout. See README 'Claude Pro/Max'." >&2
    exit 1
  fi
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  build)
    source_ros
    exec colcon build --symlink-install --paths . ranger_llm_msgs \
      --packages-select ranger_llm_msgs ranger_llm_ui
    ;;
  build-msgs)
    source_ros
    exec colcon build --symlink-install --paths . ranger_llm_msgs --packages-select ranger_llm_msgs
    ;;
  clean)
    echo "Removing build/ install/ log/ ..."
    exec rm -rf build install log
    ;;
  run)
    # Source the ROS underlay + this workspace overlay + the MobileManipulation-
    # Core overlay BEFORE launching. This is REQUIRED for the real-robot tools:
    # ranger_llm_msgs (movement actions) and manipulation_msgs come from these
    # overlays, and crucially the rosidl typesupport .so files are dlopened by
    # the RMW via LD_LIBRARY_PATH set AT PROCESS START — a sourcing this script
    # must do before `exec`, not something the Python process can fix later.
    # All sources are guarded (no-ops if absent), so pure-simulation / no-ROS
    # machines are unaffected, and `--simple` still simulates regardless.
    source_ros
    load_env
    exec python -m ranger_llm_ui.ui_node "$@"
    ;;
  run-ros)
    source_ros
    load_env
    exec ros2 launch ranger_llm_ui ranger_llm_ui.launch.py "$@"
    ;;
  proxy)
    require_cliproxy
    cd "$CLIPROXY_DIR"
    pkill -x cli-proxy-api 2>/dev/null || true
    sleep 1
    exec ./cli-proxy-api -config config.yaml
    ;;
  proxy-login)
    require_cliproxy
    cd "$CLIPROXY_DIR"
    exec ./cli-proxy-api -claude-login -no-browser -config config.yaml
    ;;
  test)
    load_env
    exec python -m pytest tests/ "$@"
    ;;
  shell)
    load_env
    source_ros
    exec "${SHELL:-bash}"
    ;;
  help|--help|-h|"")
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    echo "Run 'scripts/dev.sh help' for usage." >&2
    exit 2
    ;;
esac
