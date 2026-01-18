# Ranger LLM UI Agent Guide

## Purpose
This repo provides a Gradio-based chat UI and LangChain agent that control a Ranger robot through safe, ROS 2-compatible tools. It can run in simulation mode without ROS.

## Setup
- Python 3.10+ required; ROS 2 Humble is optional (needed for real robot integration).
- Install dependencies:
  - `pip install -r requirements.txt`
  - `pip install -e ros-technician-cli/`
  - `pip install -e .`
- Configure environment variables by copying `.env.example` to `.env` and setting provider/API keys.

## Run
- Simulation mode (no ROS 2): `python -m ranger_llm_ui.ui_node --simple`
- ROS 2 mode:
  - `ros2 run ranger_llm_ui ui_node`
  - or `ros2 launch ranger_llm_ui ranger_llm_ui.launch.py`

## Tests and Lint
- Unit tests: `pytest tests/ -v`
- Code style: `black ranger_llm_ui/`, `flake8 ranger_llm_ui/`, `mypy ranger_llm_ui/`

## Project Layout
- `ranger_llm_ui/`: core package (UI node, agent, tools, safety, schemas)
- `config/`: configuration defaults
- `launch/`: ROS 2 launch files
- `tests/`: pytest suite (simulation-safe)

## Notes for Changes
- Favor simulation-safe changes unless explicitly targeting ROS 2 hardware.
- Keep safety checks intact (velocity limits, distance validation, emergency stop).
- When adding tools, register them in `ranger_llm_ui/tools/all_tools.py` and add tests in `tests/`.
