"""
Scenario loading and parsing for the Ranger LLM UI.

A *scenario* is a plain-text "prompt file fed line-by-line": each non-empty,
non-comment line is one natural-language command sent to the agent in order,
with conversation context carried across steps (the equivalent of looping
``claude -p "$prompt" --continue`` over a ``prompts.txt`` file).

File format (``.txt`` / ``.scenario``)::

    # title: Garden Perimeter Patrol
    # description: Drive a small square patrol and report battery.
    # safety: stop
    # Lines starting with '#' are comments / metadata. Blank lines are ignored.

    Check your battery level before we start.
    Move forward 1 meter.
    Turn left 90 degrees.

Recognized metadata header keys (case-insensitive, anywhere in the comments):
``title``, ``description`` (alias ``desc``), ``safety`` (alias ``on_error``),
and ``fresh_context`` (alias ``fresh``). Everything else on a ``#`` line is an
ordinary comment.

The actual sequential execution + safety handling lives in
:mod:`ranger_llm_ui.scenario_runner`; this module is intentionally a leaf
(pure parsing + filesystem discovery) with no agent / ROS dependencies so it is
trivially testable.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Recognized scenario file extensions.
SCENARIO_EXTS = (".txt", ".scenario")

# --- Safety policies -------------------------------------------------------
# How the runner reacts when a step looks like it failed. Kept here (the leaf
# module) so both the runner and the loader's ``# safety:`` metadata agree.
POLICY_STOP = "stop"            # heuristic tripwire -> emergency stop + halt
POLICY_SUPERVISE = "supervise"  # LLM safety supervisor adjudicates / mitigates
POLICY_CONTINUE = "continue"    # log issues but never auto-halt (manual e-stop)

POLICY_LABELS = {
    POLICY_STOP: "Stop on error (safe)",
    POLICY_SUPERVISE: "AI supervisor (auto-mitigate)",
    POLICY_CONTINUE: "Run all (ignore errors)",
}
# Reverse map from the human label back to the policy key.
LABEL_TO_POLICY = {label: key for key, label in POLICY_LABELS.items()}

# Free-form metadata values -> canonical policy key.
_SAFETY_ALIASES = {
    "stop": POLICY_STOP,
    "safe": POLICY_STOP,
    "halt": POLICY_STOP,
    "supervise": POLICY_SUPERVISE,
    "supervisor": POLICY_SUPERVISE,
    "mitigate": POLICY_SUPERVISE,
    "ai": POLICY_SUPERVISE,
    "auto": POLICY_SUPERVISE,
    "continue": POLICY_CONTINUE,
    "ignore": POLICY_CONTINUE,
    "all": POLICY_CONTINUE,
    "none": POLICY_CONTINUE,
}

_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_FALSE_VALUES = {"0", "false", "no", "off", "n"}

# Metadata header: "# key: value" (also tolerates "#key:value").
_META_RE = re.compile(r"^#\s*([A-Za-z_]+)\s*:\s*(.*)$")


def policy_label(policy: str) -> str:
    """Human-readable label for a policy key (falls back to the safe default)."""
    return POLICY_LABELS.get(policy, POLICY_LABELS[POLICY_STOP])


@dataclass
class Scenario:
    """A parsed scenario: ordered prompt steps plus presentation metadata."""

    name: str                       # stable id (file stem, or "custom")
    title: str                      # display title
    description: str                # one-line description
    steps: list[str]                # ordered prompts (one per line)
    safety: str = POLICY_STOP       # default safety policy for this scenario
    fresh_context: bool = True      # clear chat history before running
    path: Optional[str] = None      # source file path (None for in-memory)
    raw: str = ""                   # original file text (for the editor box)

    @property
    def num_steps(self) -> int:
        return len(self.steps)


def _prettify(stem: str) -> str:
    """Turn a file stem like ``garden_patrol`` into ``Garden Patrol``."""
    return re.sub(r"[_\-]+", " ", stem).strip().title()


def parse_scenario_text(
    text: str,
    name: str = "custom",
    path: Optional[str] = None,
) -> Scenario:
    """Parse raw scenario text into a :class:`Scenario`.

    Metadata headers (``# key: value``) are extracted; every other non-blank,
    non-comment line becomes a step. Whitespace around steps is stripped.
    """
    title: Optional[str] = None
    description = ""
    safety = POLICY_STOP
    fresh_context = True
    steps: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = _META_RE.match(line)
            if not m:
                continue  # plain comment
            key = m.group(1).lower()
            value = m.group(2).strip()
            if key == "title":
                title = value or title
            elif key in ("description", "desc"):
                description = value
            elif key in ("safety", "on_error", "policy"):
                safety = _SAFETY_ALIASES.get(value.lower(), safety)
            elif key in ("fresh_context", "fresh", "reset_context"):
                low = value.lower()
                if low in _TRUE_VALUES:
                    fresh_context = True
                elif low in _FALSE_VALUES:
                    fresh_context = False
                # unrecognized/empty value keeps the default (mirrors `safety`)
            # unknown keys are treated as comments
            continue
        steps.append(line)

    if not title:
        title = _prettify(name)

    return Scenario(
        name=name,
        title=title,
        description=description,
        steps=steps,
        safety=safety,
        fresh_context=fresh_context,
        path=path,
        raw=text,
    )


def parse_scenario_file(path: Path) -> Scenario:
    """Parse a scenario file from disk."""
    text = path.read_text(encoding="utf-8")
    return parse_scenario_text(text, name=path.stem, path=str(path))


def scenarios_dir() -> Optional[Path]:
    """Resolve the directory holding scenario files.

    Resolution order:
      1. ``SCENARIOS_DIR`` environment variable (used exclusively if set).
      2. ``<repo-root>/scenarios`` (source checkout — repo root is the parent
         of the ``ranger_llm_ui`` package directory).
      3. ``<cwd>/scenarios``.
      4. The installed ``share/ranger_llm_ui/scenarios`` directory (colcon).

    Returns the first existing candidate, or ``None`` if none exist.
    """
    env_dir = os.getenv("SCENARIOS_DIR")
    if env_dir:
        p = Path(env_dir).expanduser()
        return p if p.is_dir() else None

    # Order matters: prefer the package-relative (source checkout) and installed
    # locations over the current working directory, so an unrelated ``scenarios/``
    # folder in some random cwd cannot hijack discovery of the bundled set.
    candidates: list[Path] = [
        Path(__file__).resolve().parent.parent / "scenarios",
        # `pip install` puts setup.py data_files under <prefix>/share/...
        Path(sys.prefix) / "share" / "ranger_llm_ui" / "scenarios",
    ]
    try:  # installed (ROS 2) share path — optional, ament may be absent
        from ament_index_python.packages import get_package_share_directory

        candidates.append(
            Path(get_package_share_directory("ranger_llm_ui")) / "scenarios"
        )
    except Exception:  # pragma: no cover - ament not installed
        pass
    # Current working directory is the lowest-priority fallback.
    candidates.append(Path.cwd() / "scenarios")

    for c in candidates:
        if c.is_dir():
            return c
    return None


def load_scenarios(directory: Optional[Path] = None) -> list[Scenario]:
    """Load and parse all scenario files from ``directory`` (sorted by title).

    Malformed individual files are logged and skipped rather than aborting the
    whole load.
    """
    directory = directory or scenarios_dir()
    if directory is None or not directory.is_dir():
        logger.info("No scenarios directory found; scenario list will be empty.")
        return []

    found: list[Scenario] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SCENARIO_EXTS or not path.is_file():
            continue
        try:
            scenario = parse_scenario_file(path)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to parse scenario %s: %s", path, e)
            continue
        if scenario.num_steps == 0:
            logger.debug("Skipping scenario with no steps: %s", path)
            continue
        found.append(scenario)

    found.sort(key=lambda s: s.title.lower())
    logger.info("Loaded %d scenario(s) from %s", len(found), directory)
    return found
