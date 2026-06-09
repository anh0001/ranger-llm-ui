"""
Manipulation Tools - High-level arm/manipulation skills for the LangChain agent.

These tools let the Ranger agent invoke the manipulation *skills* exposed by the
companion MobileManipulationCore stack. A single ROS 2 action fronts every
registered skill:

    manipulation_msgs/action/ExecuteSkill   on   /execute_skill

The skill is selected by name (``skill``) and its arguments travel as a JSON
object string (``params_json``). One action type fronts every skill so adding a
skill on the robot side needs no new message and no rebuild here. The skills
wrapped by this module mirror MobileManipulationCore's skill registry:

    pick            - grasp an object by open-vocabulary name (wrist camera)
    place           - release the held object into/onto a named receptacle
    pick_and_place  - localize destination first, then pick, then place
    arm_home_pose   - park the arm at a named pose ('rest' / 'ready')
    arm_ready_pose  - raise the arm to the look-down 'ready' capture pose
    handover        - present the held object to a person and release it

Runtime requirements (real robot only; --simple mode simulates these tools):
  * MobileManipulationCore's ``manipulation_msgs`` package built so
    ``ExecuteSkill`` can be imported. Sourcing the MMC overlay is NOT required:
    if it is built but not on PYTHONPATH (e.g. the UI was launched with a plain
    ``ros2 run`` / ``ros2 launch`` instead of scripts/dev.sh), it is
    auto-discovered under ``../MobileManipulationCore/install`` and added to
    sys.path. Override with ``MMC_INSTALL`` (the MMC ``install/`` dir) or
    ``MANIPULATION_MSGS_PYTHONPATH`` (a dist-packages dir) if it lives elsewhere.
  * MobileManipulationCore's ``skill_server`` running so /execute_skill is
    advertised. That server is brought up by ``manipulation_bringup
    core_launch.py`` on top of the ranger-garden-assistant base stack.

Sibling repo: https://github.com/anh0001/MobileManipulationCore
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Type, Any, Dict

from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from ranger_llm_ui.utils.logger import log_tool_call

logger = logging.getLogger(__name__)

# Try to import ROS 2, but allow running without it for testing
try:
    import rclpy  # noqa: F401
    from rclpy.action import ActionClient
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning("ROS 2 (rclpy) not available. Manipulation tools run in simulation mode.")

# --------------------------------------------------------------------------- #
# manipulation_msgs discovery / self-healing import
#
# The skill action type lives in MobileManipulationCore's ``manipulation_msgs``
# package. It imports cleanly once that workspace's install space is on
# PYTHONPATH: the rosidl-generated Python bindings carry an RPATH to their own
# typesupport .so files, so just adding the dist-packages dir is enough — the
# base message deps (action_msgs, builtin_interfaces, ...) come from the
# already-sourced /opt/ros/<distro>. When the UI is launched WITHOUT sourcing
# the MMC overlay (e.g. plain ``ros2 run`` / ``ros2 launch`` rather than
# scripts/dev.sh, which sources it), we self-heal by locating that dir and
# prepending it to sys.path before importing — so pick/place work regardless of
# how the UI was started, as long as MMC's manipulation_msgs has been built.
# --------------------------------------------------------------------------- #
def _find_manipulation_msgs_pythonpath() -> Optional[str]:
    """Locate the dist/site-packages dir that contains ``manipulation_msgs``.

    Honors, in order: ``MANIPULATION_MSGS_PYTHONPATH`` (a dist-packages dir),
    ``MMC_INSTALL`` (the MobileManipulationCore ``install/`` root — the same var
    scripts/dev.sh uses), then the documented sibling checkout
    ``../MobileManipulationCore/install``. Returns ``None`` if not found.
    """
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"

    def _has_pkg(d: Path) -> bool:
        return (d / "manipulation_msgs" / "__init__.py").is_file()

    # 1. Direct dist-packages override (highest priority).
    direct = os.getenv("MANIPULATION_MSGS_PYTHONPATH")
    if direct and _has_pkg(Path(direct)):
        return direct

    # 2/3. Candidate install roots: explicit MMC_INSTALL, then sibling default.
    roots: list[Path] = []
    mmc_install = os.getenv("MMC_INSTALL")
    if mmc_install:
        roots.append(Path(mmc_install))
    # this file: <repo>/ranger_llm_ui/tools/manipulation_tools.py
    #            -> repo root at parents[2], MMC is a sibling of the repo.
    try:
        repo_root = Path(__file__).resolve().parents[2]
        roots.append(repo_root.parent / "MobileManipulationCore" / "install")
    except IndexError:  # pragma: no cover - defensive
        pass

    for root in roots:
        if not root.is_dir():
            continue
        # Cover colcon isolated install (per-package prefix) and merged install
        # layouts, with both local/lib and lib, dist- and site-packages.
        for base in (root / "manipulation_msgs", root):
            for libdir in ("local/lib", "lib"):
                for pkgs in ("dist-packages", "site-packages"):
                    cand = base / libdir / pyver / pkgs
                    if _has_pkg(cand):
                        return str(cand)
        # Fallback: bounded glob in case the layout differs from the above.
        try:
            for init in root.glob(
                f"**/{pyver}/*-packages/manipulation_msgs/__init__.py"
            ):
                return str(init.parent.parent)
        except OSError:  # pragma: no cover - best effort
            pass
    return None


# Try the direct import first (works when the MMC overlay is sourced).
try:
    from manipulation_msgs.action import ExecuteSkill
    SKILL_ACTION_AVAILABLE = True
except ImportError:
    SKILL_ACTION_AVAILABLE = False

# Not sourced? Auto-discover the built manipulation_msgs and retry.
if not SKILL_ACTION_AVAILABLE and ROS_AVAILABLE:
    _mm_pythonpath = _find_manipulation_msgs_pythonpath()
    if _mm_pythonpath and _mm_pythonpath not in sys.path:
        sys.path.insert(0, _mm_pythonpath)
        try:
            from manipulation_msgs.action import ExecuteSkill
            SKILL_ACTION_AVAILABLE = True
            logger.info(
                "manipulation_msgs auto-discovered and added to sys.path: %s",
                _mm_pythonpath,
            )
        except ImportError:  # pragma: no cover - path found but import failed
            logger.debug(
                "manipulation_msgs still not importable after adding %s to sys.path",
                _mm_pythonpath,
            )

if not SKILL_ACTION_AVAILABLE and ROS_AVAILABLE:
    logger.warning(
        "manipulation_msgs not available. Manipulation skills will be inert "
        "until the MobileManipulationCore workspace is built (and reachable). "
        "Auto-discovery looked under MANIPULATION_MSGS_PYTHONPATH, MMC_INSTALL, "
        "and ../MobileManipulationCore/install. Build it there, or set "
        "MMC_INSTALL to your MMC install/ dir if it lives elsewhere. See "
        "https://github.com/anh0001/MobileManipulationCore"
    )


# Action name (override with EXECUTE_SKILL_ACTION if remapped on the robot)
EXECUTE_SKILL_ACTION = os.getenv("EXECUTE_SKILL_ACTION", "/execute_skill")

# How long to wait for the skill_server to appear before giving up.
ACTION_SERVER_WAIT_TIMEOUT_S = 5.0
# Default ceiling for a whole skill to finish, when the caller gives no
# per-skill timeout. Manipulation skills (detect -> align -> grasp -> lift) are
# slow, so this is generous. The robot-side skill applies its own timeout too.
# Parsed defensively: a malformed override must not take down the whole tool
# stack at import (this module is imported transitively with StopRobot et al.).
try:
    SKILL_GOAL_TIMEOUT_S = float(os.getenv("SKILL_GOAL_TIMEOUT_S", "300.0"))
except ValueError:
    logger.warning(
        "Invalid SKILL_GOAL_TIMEOUT_S=%r; falling back to 300.0",
        os.getenv("SKILL_GOAL_TIMEOUT_S"),
    )
    SKILL_GOAL_TIMEOUT_S = 300.0


class ManipulationInterface:
    """
    Singleton interface to MobileManipulationCore's skill server.

    Owns one ActionClient for ``/execute_skill`` and exposes a generic
    :meth:`execute_skill` that dispatches any registered skill by name. Mirrors
    the design of ``movement_tools.ROSInterface`` (shared node, deadlock-safe
    futures, cancellable active goal) so it composes with the same
    MultiThreadedExecutor that spins the UI node.
    """

    _instance: Optional["ManipulationInterface"] = None
    _node: Optional[Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, node: Optional[Any] = None):
        """Initialize the interface with a ROS node (or None for simulation)."""
        # Allow re-initialization when we previously started in simulation mode
        # and a real ROS node becomes available later.
        if self._initialized and not (self._simulation_mode and node is not None):
            return

        self._node = node
        self._simulation_mode = not ROS_AVAILABLE or node is None
        self._client = None
        self._active_goal_handle = None
        self._goal_lock = threading.Lock()

        if not self._simulation_mode and self._node is not None:
            if SKILL_ACTION_AVAILABLE:
                # NOTE: importing manipulation_msgs (so SKILL_ACTION_AVAILABLE is
                # True) is NOT sufficient to build the action client. The RMW
                # dlopens the *introspection* typesupport .so at ActionClient
                # creation via LD_LIBRARY_PATH / the ament index, NOT via the
                # Python bindings' RPATH. When MMC's Python package is reachable
                # (e.g. auto-discovered onto sys.path) but its overlay was never
                # sourced, that .so can't be found and rclpy raises ValueError
                # ("type_support is null"). Catch it so manipulation degrades to
                # inert instead of crashing agent init (and the SimpleAgent
                # fallback). Source the MMC overlay to actually enable it.
                try:
                    self._client = ActionClient(
                        self._node, ExecuteSkill, EXECUTE_SKILL_ACTION
                    )
                    logger.info(
                        "Manipulation interface initialized (action=%s)",
                        EXECUTE_SKILL_ACTION,
                    )
                except Exception as e:  # typically ValueError: type_support is null
                    self._client = None
                    logger.warning(
                        "Manipulation interface initialized WITHOUT a skill "
                        "action client: could not create the %s ActionClient "
                        "(%s). manipulation_msgs imports, but its typesupport "
                        "library isn't loadable in this process. SOURCE the "
                        "MobileManipulationCore overlay before launching "
                        "(e.g. `source ../MobileManipulationCore/install/"
                        "setup.bash`) so LD_LIBRARY_PATH/AMENT_PREFIX_PATH "
                        "include manipulation_msgs, or disable arm tools with "
                        "ENABLE_MANIPULATION_TOOLS=false.",
                        EXECUTE_SKILL_ACTION,
                        e,
                    )
            else:
                logger.warning(
                    "Manipulation interface initialized WITHOUT a skill action "
                    "client (manipulation_msgs not available). Build/source the "
                    "MobileManipulationCore workspace to enable manipulation."
                )
        else:
            logger.info("Manipulation interface running in simulation mode")

        self._initialized = True

    @property
    def simulation_mode(self) -> bool:
        # Default to simulation when not yet initialized, so a tool invoked
        # before initialize() degrades safely instead of raising.
        return getattr(self, "_simulation_mode", True)

    @property
    def action_available(self) -> bool:
        """True if a real skill action client is wired up."""
        return getattr(self, "_client", None) is not None

    @staticmethod
    def _wait_for_future(future, timeout_s: float) -> bool:
        """
        Wait for an rclpy future using a threading Event.

        Avoids rclpy.spin_until_future_complete(), which deadlocks when the node
        is already spun by a MultiThreadedExecutor. Returns True if completed.
        """
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        return event.wait(timeout=timeout_s)

    def cancel_active_goal(self):
        """Cancel the currently active skill goal, if any."""
        with self._goal_lock:
            if self._active_goal_handle is not None:
                logger.info("Cancelling active manipulation skill goal")
                try:
                    self._active_goal_handle.cancel_goal_async()
                except Exception as e:  # pragma: no cover - best effort
                    logger.warning(f"Error cancelling skill goal: {e}")
                self._active_goal_handle = None

    def execute_skill(
        self,
        skill: str,
        params: Optional[Dict[str, Any]] = None,
        timeout_s: float = SKILL_GOAL_TIMEOUT_S,
    ) -> tuple[bool, str]:
        """
        Send an ExecuteSkill goal and block until the skill completes.

        Args:
            skill: registered skill name (e.g. "pick", "place", "arm_home_pose").
            params: skill arguments; serialized to the goal's params_json.
            timeout_s: max seconds to wait for the result before giving up.

        Returns:
            (success, message): the authoritative outcome boolean (from the
            action result, or False for any client-side error/timeout) and a
            human-readable status string for the operator/LLM. Any structured
            result_json is appended inline as "(details: ...)".
        """
        success, message, raw = self._send_and_wait(skill, params, timeout_s)
        details = (raw or "").strip()
        if details and details not in ("{}", "null"):
            message += f" (details: {details})"
        return success, message

    def execute_skill_detailed(
        self,
        skill: str,
        params: Optional[Dict[str, Any]] = None,
        timeout_s: float = SKILL_GOAL_TIMEOUT_S,
    ) -> tuple[bool, str, Dict[str, Any]]:
        """Like :meth:`execute_skill` but also returns the parsed result_json.

        For skills whose outputs the caller wants to format itself (e.g.
        localize_object returns object poses). Returns (success, message, data)
        where ``data`` is the decoded result_json object ({} if absent/invalid).
        """
        success, message, raw = self._send_and_wait(skill, params, timeout_s)
        data: Dict[str, Any] = {}
        s = (raw or "").strip()
        if s and s not in ("null",):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    data = parsed
            except (json.JSONDecodeError, ValueError):
                data = {}
        return success, message, data

    def _send_and_wait(
        self,
        skill: str,
        params: Optional[Dict[str, Any]] = None,
        timeout_s: float = SKILL_GOAL_TIMEOUT_S,
    ) -> tuple[bool, str, str]:
        """Send a goal, block for the result, and return (success, message, raw_json).

        ``raw_json`` is the result's result_json string ("" on any client-side
        error/timeout). Shared by execute_skill (which inlines the details) and
        execute_skill_detailed (which parses them).
        """
        if self._client is None:
            return False, (
                "Error: ExecuteSkill action client not available because "
                "manipulation_msgs could not be imported. Build the "
                "MobileManipulationCore manipulation_msgs package; it is then "
                "auto-discovered under ../MobileManipulationCore/install. If it "
                "lives elsewhere, set MMC_INSTALL (its install/ dir) or "
                "MANIPULATION_MSGS_PYTHONPATH before launching the UI."
            ), ""

        # Wait for the skill server
        if not self._client.wait_for_server(timeout_sec=ACTION_SERVER_WAIT_TIMEOUT_S):
            return False, (
                f"Error: {EXECUTE_SKILL_ACTION} action server not available. "
                "Ensure the MobileManipulationCore skill_server is running "
                "(ros2 launch manipulation_bringup core_launch.py)."
            ), ""

        # Build goal
        goal_msg = ExecuteSkill.Goal()
        goal_msg.skill = str(skill)
        goal_msg.params_json = json.dumps(params or {})

        logger.info("Sending skill goal: %s params=%s", skill, goal_msg.params_json)

        def _on_feedback(feedback_msg):
            fb = feedback_msg.feedback
            try:
                pct = (fb.progress or 0.0) * 100.0
            except Exception:
                pct = 0.0
            logger.info("[skill %s] %s (%.0f%%)", skill, fb.state, pct)

        # Send goal (resolved by the executor's background thread)
        send_future = self._client.send_goal_async(
            goal_msg, feedback_callback=_on_feedback
        )
        if not self._wait_for_future(send_future, timeout_s=5.0):
            return False, f"Error: Timed out waiting for skill '{skill}' goal to be accepted", ""

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, f"Error: Skill '{skill}' goal was rejected by the skill server", ""

        # Track active goal for cancellation
        with self._goal_lock:
            self._active_goal_handle = goal_handle

        # Wait for the result
        result_future = goal_handle.get_result_async()
        completed = self._wait_for_future(result_future, timeout_s=timeout_s)

        # Clear active goal
        with self._goal_lock:
            self._active_goal_handle = None

        if not completed or result_future.result() is None:
            try:
                goal_handle.cancel_goal_async()
            except Exception:  # pragma: no cover - best effort
                pass
            return False, f"Error: Skill '{skill}' timed out after {timeout_s:.0f}s (cancel requested)", ""

        result = result_future.result().result
        status = "succeeded" if result.success else "failed"
        message = (result.message or "").strip()
        out = f"Skill '{skill}' {status}"
        if message:
            out += f": {message}"
        return bool(result.success), out, (result.result_json or "")


# Global manipulation interface instance
_manip_interface: Optional[ManipulationInterface] = None


def get_manipulation_interface() -> ManipulationInterface:
    """Get or create the manipulation interface singleton."""
    global _manip_interface
    if _manip_interface is None:
        _manip_interface = ManipulationInterface()
    return _manip_interface


def initialize_manipulation_interface(node: Optional[Any] = None) -> ManipulationInterface:
    """Initialize the manipulation interface with a node."""
    interface = get_manipulation_interface()
    interface.initialize(node)
    return interface


def _skill_timeout(timeout_sec: float, phases: int = 1) -> float:
    """Client-side result-wait ceiling derived from the skill's own timeout.

    The skill's ``timeout_sec`` (0 = server default) bounds the robot-side
    attempt; the client must wait at least as long plus slack. Falls back to the
    module default when the caller leaves it at 0.
    """
    if timeout_sec and timeout_sec > 0:
        return float(timeout_sec) * max(1, phases) + 60.0
    return SKILL_GOAL_TIMEOUT_S * max(1, phases)


# --------------------------------------------------------------------------- #
# Pydantic argument schemas (mirror each skill's declared params)
# --------------------------------------------------------------------------- #
class PickInput(BaseModel):
    """Input schema for the Pick tool."""
    object: str = Field(
        description=(
            'Open-vocabulary label of the thing to grasp, e.g. "bread", '
            '"red apple", "toy banana". For toy fruits prefix with "toy".'
        )
    )
    timeout_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Max seconds for the whole attempt; 0 uses the server default.",
    )


class PlaceInput(BaseModel):
    """Input schema for the Place tool."""
    target: str = Field(
        description=(
            'Open-vocabulary label of where to place the held object, e.g. '
            '"box", "table", "bowl", "plate".'
        )
    )
    timeout_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Max seconds for the whole attempt; 0 uses the server default.",
    )


class PickAndPlaceInput(BaseModel):
    """Input schema for the PickAndPlace tool."""
    object: str = Field(
        description='Label of the object to pick, e.g. "banana", "toy banana", "bread".'
    )
    destination: str = Field(
        description=(
            'Label of where to place it, e.g. "white box", "box", "plate". '
            "Localized first while the gripper is empty so the held object "
            "never blocks detection at place time."
        )
    )
    timeout_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Max seconds for EACH phase; 0 uses the server default.",
    )


class HomeArmInput(BaseModel):
    """Input schema for the HomeArm tool."""
    pose: str = Field(
        default="rest",
        description=(
            "Named arm pose: 'rest' (default; folds the arm to all-zero joints "
            "/ park, lifting the wrist clear of the LiDAR first) or 'ready' "
            "(look-down capture pose used before a pick — prefer the ReadyArm "
            "tool for that)."
        ),
    )
    time_sec: float = Field(
        default=5.0,
        gt=0.0,
        description="Seconds to take for the move.",
    )


class ReadyArmInput(BaseModel):
    """Input schema for the ReadyArm tool."""
    time_sec: float = Field(
        default=5.0,
        gt=0.0,
        description="Seconds to take for the move.",
    )


class HandoverInput(BaseModel):
    """Input schema for the Handover tool."""
    dwell_sec: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Seconds to hold the object out before opening the gripper; "
            "0 uses the server default."
        ),
    )
    posture: str = Field(
        default="auto",
        description=(
            "Recipient posture, sets the present height: 'auto' (default, "
            "picked from detected head height), 'standing' (forces high pose), "
            "or 'seated' (forces low pose). Use standing/seated when known."
        ),
    )
    timeout_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Reserved; the handover is bounded by its own move timeouts.",
    )


class PickAtInput(BaseModel):
    """Input schema for the PickAt tool."""
    x: float = Field(description="X (forward, metres) of the grasp point.")
    y: float = Field(description="Y (left, metres) of the grasp point.")
    z: float = Field(description="Z (up, metres) of the grasp point.")
    frame: str = Field(
        default="base_footprint",
        description="TF frame of the point (default base_footprint — e.g. a point "
                    "from LocateObject).",
    )


class PlaceAtInput(BaseModel):
    """Input schema for the PlaceAt tool."""
    x: float = Field(description="X (forward, metres) of the release point.")
    y: float = Field(description="Y (left, metres) of the release point.")
    z: float = Field(description="Z (up, metres) of the release point.")
    frame: str = Field(
        default="base_footprint",
        description="TF frame of the point (default base_footprint).",
    )


class PickAndPlaceAtInput(BaseModel):
    """Input schema for the PickAndPlaceAt tool."""
    pick_x: float = Field(description="X (forward, m) of the grasp point.")
    pick_y: float = Field(description="Y (left, m) of the grasp point.")
    pick_z: float = Field(description="Z (up, m) of the grasp point.")
    place_x: float = Field(description="X (forward, m) of the release point.")
    place_y: float = Field(description="Y (left, m) of the release point.")
    place_z: float = Field(description="Z (up, m) of the release point.")
    frame: str = Field(
        default="base_footprint",
        description="TF frame of both points (default base_footprint).",
    )


class LookAtInput(BaseModel):
    """Input schema for the LookAt tool."""
    x: float = Field(description="X (forward, metres) of the point to look at.")
    y: float = Field(description="Y (left, metres) of the point to look at.")
    z: float = Field(description="Z (up, metres) of the point to look at.")
    frame: str = Field(
        default="base_footprint",
        description="TF frame of the point (default base_footprint).",
    )
    standoff_m: float = Field(
        default=0.0,
        ge=0.0,
        description="If > 0, first move the wrist camera to this distance (metres) "
                    "from the point, then aim at it. 0 (default) only re-aims from "
                    "the current camera position.",
    )


class LocateObjectInput(BaseModel):
    """Input schema for the LocateObject tool."""
    objects: str = Field(
        description=(
            'One or more object labels to locate, comma-separated, e.g. "banana" '
            'or "banana, red cup, bottle". Open-vocabulary.'
        )
    )
    camera: str = Field(
        default="wrist",
        description=(
            'Which depth camera to look through: "wrist" (the D405 on the arm — '
            "most accurate, but the object must be in the wrist camera's view, so "
            'raise the arm with ReadyArm first if it is parked) or "rear" (the '
            "fixed D435i behind the arm)."
        ),
    )


def _format_localization(data: Dict[str, Any], fallback_message: str) -> str:
    """Render the localize_object result_json into a concise answer for the LLM."""
    objects = data.get("objects") if isinstance(data, dict) else None
    if not objects:
        # No structured payload (skill failed early / server error): pass the
        # human-readable message straight through.
        return fallback_message
    frame = data.get("frame", "base_footprint")
    camera = data.get("camera", "?")
    calib = str(data.get("calibration", ""))
    lines = []
    for o in objects:
        label = o.get("label", "object")
        if o.get("found"):
            p = o.get("position", {}) or {}
            x, y, z = p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)
            extra = []
            if o.get("distance_m") is not None:
                extra.append(f"{o['distance_m']:.2f} m away")
            if o.get("depth_m") is not None:
                extra.append(f"depth {o['depth_m']:.2f} m")
            if o.get("score"):
                extra.append(f"confidence {o['score']:.2f}")
            suffix = f" ({', '.join(extra)})" if extra else ""
            lines.append(
                f"- {label}: x={x:.2f}, y={y:.2f}, z={z:.2f} m in {frame}{suffix}")
        else:
            lines.append(f"- {label}: not found ({o.get('reason', 'not detected')})")
    out = f"Object positions ({camera} camera, frame {frame}):\n" + "\n".join(lines)
    if calib.startswith("extrinsic") and "UNCALIBRATED" in calib:
        out += ("\nNote: the rear camera's mounting extrinsic is not yet calibrated, "
                "so these rear-camera coordinates are approximate.")
    return out


def _simulate_localization(labels: list, camera: str) -> str:
    """Plausible simulated object positions for --simple mode (no robot)."""
    if not labels:
        return "No objects requested."
    lines = []
    for i, label in enumerate(labels):
        x = round(0.40 + 0.05 * i, 2)
        y = round(-0.10 + 0.08 * i, 2)
        z = 0.15
        lines.append(
            f"- {label}: x={x:.2f}, y={y:.2f}, z={z:.2f} m in base_footprint")
    return (f"Object positions ({camera} camera, frame base_footprint) [SIMULATED]:\n"
            + "\n".join(lines))


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
class _SkillToolBase(BaseTool):
    """Shared plumbing for skill-backed tools (sim handling + logging)."""

    return_direct: bool = False

    class Config:
        arbitrary_types_allowed = True

    # Subclasses set these:
    skill_name: str = ""

    def _dispatch(
        self,
        params: Dict[str, Any],
        timeout_s: float,
        log_params: Dict[str, Any],
    ) -> str:
        """Run the skill (or simulate it) and emit a tool-call log entry."""
        start_time = time.time()
        manip = get_manipulation_interface()

        if manip.simulation_mode:
            logger.info("[SIM] %s skill params=%s", self.skill_name, params)
            result = f"Executed '{self.skill_name}' with {params} (simulated)"
            log_tool_call(
                tool_name=self.name,
                parameters=log_params,
                result=result,
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return result

        try:
            success, result = manip.execute_skill(
                self.skill_name, params, timeout_s=timeout_s
            )
        except Exception as e:
            log_tool_call(
                tool_name=self.name,
                parameters=log_params,
                result=None,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return f"Error running {self.skill_name}: {e}"

        log_tool_call(
            tool_name=self.name,
            parameters=log_params,
            result=result,
            success=success,
            execution_time_ms=(time.time() - start_time) * 1000,
        )
        return result


class PickTool(_SkillToolBase):
    """Pick an object by name with the robot arm."""

    name: str = "Pick"
    skill_name: str = "pick"
    description: str = (
        "Pick a single object by open-vocabulary name with the robot arm. Runs "
        "the full wrist-camera visual-servo grasp: detect -> align -> grasp -> "
        "lift. Use for 'pick up the bread', 'grab the red apple'. Succeeds only "
        "when the post-lift gripper width indicates the object is actually held; "
        "a full-close (empty) grasp is reported as a likely-empty failure. After "
        "a successful pick, use Place or Handover to set the object down. Input: "
        "object (label), optional timeout_sec."
    )
    args_schema: Type[BaseModel] = PickInput

    def _run(
        self,
        object: str,
        timeout_sec: float = 0.0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        params: Dict[str, Any] = {"object": object}
        if timeout_sec and timeout_sec > 0:
            params["timeout_sec"] = float(timeout_sec)
        return self._dispatch(
            params,
            timeout_s=_skill_timeout(timeout_sec),
            log_params={"object": object, "timeout_sec": timeout_sec},
        )


class PlaceTool(_SkillToolBase):
    """Place the currently-held object onto/into a named target."""

    name: str = "Place"
    skill_name: str = "place"
    description: str = (
        "Place the object the robot is currently holding into/onto a detected "
        "target. Names a receptacle to detect (e.g. 'box', 'table', 'bowl'), "
        "runs the visual-servo pipeline to localize it, approaches the standoff "
        "above it, and opens the gripper to release. RUN ONLY AFTER A SUCCESSFUL "
        "Pick. Input: target (label), optional timeout_sec."
    )
    args_schema: Type[BaseModel] = PlaceInput

    def _run(
        self,
        target: str,
        timeout_sec: float = 0.0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        params: Dict[str, Any] = {"target": target}
        if timeout_sec and timeout_sec > 0:
            params["timeout_sec"] = float(timeout_sec)
        return self._dispatch(
            params,
            timeout_s=_skill_timeout(timeout_sec),
            log_params={"target": target, "timeout_sec": timeout_sec},
        )


class PickAndPlaceTool(_SkillToolBase):
    """Pick an object and place it onto/into a destination in one call."""

    name: str = "PickAndPlace"
    skill_name: str = "pick_and_place"
    description: str = (
        "Pick an object and place it into/onto a destination in one call. The "
        "destination is localized FIRST while the gripper is empty and the wrist "
        "camera is unobstructed, then reused for the place — so the held object "
        "never blocks detection. Aborts before picking if either the object or "
        "destination is not detected up front. PREFER THIS over separate Pick + "
        "Place when both the object and its destination are known (e.g. 'put the "
        "banana in the white box'). Input: object, destination, optional timeout_sec."
    )
    args_schema: Type[BaseModel] = PickAndPlaceInput

    def _run(
        self,
        object: str,
        destination: str,
        timeout_sec: float = 0.0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        params: Dict[str, Any] = {"object": object, "destination": destination}
        if timeout_sec and timeout_sec > 0:
            params["timeout_sec"] = float(timeout_sec)
        return self._dispatch(
            params,
            # two phases (pick + place), each bounded by timeout_sec
            timeout_s=_skill_timeout(timeout_sec, phases=2),
            log_params={
                "object": object,
                "destination": destination,
                "timeout_sec": timeout_sec,
            },
        )


class HomeArmTool(_SkillToolBase):
    """Park the arm at a named home pose ('rest' or 'ready')."""

    name: str = "HomeArm"
    skill_name: str = "arm_home_pose"
    description: str = (
        "Move the robot arm to a named home/park pose. Defaults to 'rest', the "
        "all-zero joint pose — folding to it first lifts the wrist up to clear "
        "the Mid-360 LiDAR. Use it to park the arm safely between tasks. 'ready' "
        "(the look-down capture pose used before a pick) is also selectable, but "
        "PREFER the dedicated ReadyArm tool for that. Input: pose ('rest' or "
        "'ready'), optional time_sec."
    )
    args_schema: Type[BaseModel] = HomeArmInput

    def _run(
        self,
        pose: str = "rest",
        time_sec: float = 5.0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        params: Dict[str, Any] = {"pose": pose, "time_sec": float(time_sec)}
        # home is quick; wait a bit beyond the requested move time.
        timeout_s = max(30.0, float(time_sec) + 20.0)
        return self._dispatch(
            params,
            timeout_s=timeout_s,
            log_params={"pose": pose, "time_sec": time_sec},
        )


class ReadyArmTool(_SkillToolBase):
    """Raise the arm to the look-down 'ready' capture pose."""

    name: str = "ReadyArm"
    skill_name: str = "arm_ready_pose"
    description: str = (
        "Move the robot arm to the 'ready' pose: the look-down capture pose used "
        "before a pick. Use it to raise the arm from rest/zero into a working "
        "configuration. The pose stays in sync with the pick/handover skills. "
        "Input: optional time_sec."
    )
    args_schema: Type[BaseModel] = ReadyArmInput

    def _run(
        self,
        time_sec: float = 5.0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        params: Dict[str, Any] = {"time_sec": float(time_sec)}
        # ready is a quick single move; wait a bit beyond the requested time.
        timeout_s = max(30.0, float(time_sec) + 20.0)
        return self._dispatch(
            params,
            timeout_s=timeout_s,
            log_params={"time_sec": time_sec},
        )


class HandoverTool(_SkillToolBase):
    """Hand the currently-held object to a person."""

    name: str = "Handover"
    skill_name: str = "handover"
    description: str = (
        "Hand the currently-held object to a person. Uses the base-mounted "
        "camera to find the nearest person and their distance, yaws the arm "
        "toward them, presents the object at a safe distance, dwells so they can "
        "take it, then opens the gripper. RUN ONLY AFTER A SUCCESSFUL Pick. "
        "Aborts if no object is held, no clear single person is seen, or the "
        "person is too close / too far / too far to the side. Input: optional "
        "dwell_sec, posture ('auto'/'standing'/'seated'), timeout_sec."
    )
    args_schema: Type[BaseModel] = HandoverInput

    def _run(
        self,
        dwell_sec: float = 0.0,
        posture: str = "auto",
        timeout_sec: float = 0.0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        params: Dict[str, Any] = {"posture": posture}
        if dwell_sec and dwell_sec > 0:
            params["dwell_sec"] = float(dwell_sec)
        if timeout_sec and timeout_sec > 0:
            params["timeout_sec"] = float(timeout_sec)
        return self._dispatch(
            params,
            timeout_s=_skill_timeout(timeout_sec),
            log_params={
                "dwell_sec": dwell_sec,
                "posture": posture,
                "timeout_sec": timeout_sec,
            },
        )


class PickAtTool(_SkillToolBase):
    """Grasp at a given 3D point (no detection)."""

    name: str = "PickAt"
    skill_name: str = "pick_at"
    description: str = (
        "Grasp at a GIVEN 3D point with the arm — no detection. I open the gripper, "
        "descend onto the point from a standoff above it, and close. The point is "
        "(x, y, z) in metres in base_footprint by default (x forward, y left, z up) "
        "— typically a point from LocateObject. CHAIN: call LocateObject first to "
        "get the object's position, then PickAt with that x/y/z. I trust the "
        "coordinate (no detection), so only use a point from LocateObject or a "
        "measured one. Succeeds only if something is actually held after the lift. "
        "Input: x, y, z, optional frame."
    )
    args_schema: Type[BaseModel] = PickAtInput

    def _run(self, x: float, y: float, z: float, frame: str = "base_footprint",
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        params: Dict[str, Any] = {
            "position": [float(x), float(y), float(z)], "frame": frame}
        return self._dispatch(params, timeout_s=_skill_timeout(0.0), log_params=params)


class PlaceAtTool(_SkillToolBase):
    """Release the held object at a given 3D point (no detection)."""

    name: str = "PlaceAt"
    skill_name: str = "place_at"
    description: str = (
        "Release the object I am holding at a GIVEN 3D point — no detection. I "
        "approach a standoff above the point and open the gripper to drop the "
        "object there. The point is (x, y, z) in metres in base_footprint by "
        "default. RUN ONLY AFTER a successful Pick or PickAt (I must be holding "
        "something). Input: x, y, z, optional frame."
    )
    args_schema: Type[BaseModel] = PlaceAtInput

    def _run(self, x: float, y: float, z: float, frame: str = "base_footprint",
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        params: Dict[str, Any] = {
            "position": [float(x), float(y), float(z)], "frame": frame}
        return self._dispatch(params, timeout_s=_skill_timeout(0.0), log_params=params)


class PickAndPlaceAtTool(_SkillToolBase):
    """Grasp at one given 3D point and release at another (no detection)."""

    name: str = "PickAndPlaceAt"
    skill_name: str = "pick_and_place_at"
    description: str = (
        "Grasp at one GIVEN 3D point and release at another — no detection. I pick "
        "at (pick_x, pick_y, pick_z), lift, then release at (place_x, place_y, "
        "place_z). All in metres in base_footprint by default. Use when I already "
        "know both coordinates (e.g. from LocateObject). Input: pick_x/y/z, "
        "place_x/y/z, optional frame."
    )
    args_schema: Type[BaseModel] = PickAndPlaceAtInput

    def _run(self, pick_x: float, pick_y: float, pick_z: float,
             place_x: float, place_y: float, place_z: float,
             frame: str = "base_footprint",
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        params: Dict[str, Any] = {
            "pick_position": [float(pick_x), float(pick_y), float(pick_z)],
            "place_position": [float(place_x), float(place_y), float(place_z)],
            "frame": frame,
        }
        return self._dispatch(
            params, timeout_s=_skill_timeout(0.0, phases=2), log_params=params)


class LocateObjectTool(_SkillToolBase):
    """Report the 3D position of named objects relative to base_footprint.

    Perception tool (object localization), skill-backed: it drives MMC's
    ``localize_object`` skill over the same /execute_skill action as the
    manipulation tools, but parses the returned poses and formats them itself.
    """

    name: str = "LocateObject"
    skill_name: str = "localize_object"
    description: str = (
        "Find WHERE one or more named objects are in 3D and report each object's "
        "(x, y, z) position in metres relative to my base (base_footprint). This "
        "locates OBJECTS in the scene; it is NOT robot self-localization (knowing "
        "where I am on a map) — for my own pose use GetOdometry. Detects each "
        "object by open-vocabulary name through a depth camera: camera='wrist' "
        "(the D405 on my arm — most accurate, but the object must be in the wrist "
        "camera's view, so raise the arm with ReadyArm first if it is parked) or "
        "camera='rear' (the fixed D435i behind my arm). Handles multiple objects "
        "in one call (comma-separated). Use for 'where is the banana?', 'what is "
        "the 3D position of the red cup and the bottle from the rear camera?'. "
        "Input: objects (labels), camera ('wrist' or 'rear')."
    )
    args_schema: Type[BaseModel] = LocateObjectInput

    def _run(
        self,
        objects: str,
        camera: str = "wrist",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        start_time = time.time()
        manip = get_manipulation_interface()
        labels = [o.strip() for o in str(objects).replace(";", ",").split(",")
                  if o.strip()]
        cam = (camera or "wrist").strip().lower()
        log_params = {"objects": objects, "camera": cam}

        if manip.simulation_mode:
            result = _simulate_localization(labels, cam)
            log_tool_call(
                tool_name=self.name,
                parameters=log_params,
                result=result,
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return result

        params: Dict[str, Any] = {
            "objects": ", ".join(labels) if labels else str(objects),
            "camera": cam,
        }
        try:
            success, message, data = manip.execute_skill_detailed(
                self.skill_name, params, timeout_s=_skill_timeout(0.0)
            )
        except Exception as e:
            log_tool_call(
                tool_name=self.name,
                parameters=log_params,
                result=None,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return f"Error locating objects: {e}"

        result = _format_localization(data, message)
        log_tool_call(
            tool_name=self.name,
            parameters=log_params,
            result=result,
            success=success,
            execution_time_ms=(time.time() - start_time) * 1000,
        )
        return result


class LookAtTool(_SkillToolBase):
    """Aim the wrist camera at a 3D point so it can see that point."""

    name: str = "LookAt"
    skill_name: str = "look_at"
    description: str = (
        "Aim my wrist camera (the D405 on the arm) at a GIVEN 3D point so that "
        "point comes into the wrist camera's view. This is the precondition for "
        "LocateObject with camera='wrist' on a target that isn't already framed "
        "by the default raised pose. The point is (x, y, z) in metres in "
        "base_footprint by default (x forward, y left, z up). By default I only "
        "re-orient the camera from where it is; pass standoff_m > 0 to first move "
        "the camera that far from the point (use it when the point is out of the "
        "wrist camera's reach). CHAIN: LookAt(x, y, z) then "
        "LocateObject(camera='wrist'). If the arm is parked low, ReadyArm first. "
        "Input: x, y, z, optional frame, optional standoff_m."
    )
    args_schema: Type[BaseModel] = LookAtInput

    def _run(self, x: float, y: float, z: float, frame: str = "base_footprint",
             standoff_m: float = 0.0,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        params: Dict[str, Any] = {
            "position": [float(x), float(y), float(z)],
            "frame": frame,
            "standoff_m": float(standoff_m),
        }
        return self._dispatch(params, timeout_s=_skill_timeout(0.0), log_params=params)


# Convenience function to create all manipulation tools
def get_manipulation_tools() -> list[BaseTool]:
    """Get all manipulation-related tools (one per MMC skill).

    LookAt is included here (not in the perception tools) because it ACTUATES the
    arm — so it is enabled/disabled together with the other arm tools via
    ENABLE_MANIPULATION_TOOLS, even though it serves perception (aiming the wrist
    camera before LocateObject).
    """
    return [
        PickTool(),
        PlaceTool(),
        PickAndPlaceTool(),
        PickAtTool(),
        PlaceAtTool(),
        PickAndPlaceAtTool(),
        HomeArmTool(),
        ReadyArmTool(),
        HandoverTool(),
        LookAtTool(),
    ]


def get_perception_skill_tools() -> list[BaseTool]:
    """Skill-backed perception tools (object localization via the skill server).

    Note: LookAt is a perception *precondition* but lives with the manipulation
    tools (see get_manipulation_tools) because it moves the arm.
    """
    return [
        LocateObjectTool(),
    ]
