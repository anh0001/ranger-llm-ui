# Ranger Scenarios

A **scenario** is a plain-text *prompt file fed line-by-line*: each non-empty,
non-comment line is one natural-language command sent to the robot **in order**,
with the conversation context carried across steps. It is the UI equivalent of
the classic shell loop:

```bash
while IFS= read -r prompt; do
  claude -p "$prompt" --model claude-sonnet-4-6 --continue
done < prompts.txt
```

Select, preview, edit, and run scenarios from the **Scenarios** tab of the web
UI (`http://localhost:7860`). Every step streams into a live transcript with a
progress bar, and a safety net stops or recovers the robot when a step goes
wrong.

## File format

Save scenarios as `*.txt` (or `*.scenario`) in this directory:

```
# title: Garden Perimeter Patrol
# description: Drive a small square patrol and report battery.
# safety: stop
# fresh_context: true
#
# '#' lines are comments/metadata. Blank lines are ignored.

Check your battery level before we start.
Move forward 1 meter.
Turn left 90 degrees.
```

### Metadata headers (optional, case-insensitive)

| Key                         | Meaning                                                            | Default            |
| --------------------------- | ----------------------------------------------------------------- | ------------------ |
| `title`                     | Display name in the picker                                         | prettified filename |
| `description` (`desc`)      | One-line summary                                                   | empty              |
| `safety` (`on_error`)       | Default safety policy: `stop`, `supervise`, or `continue`         | `stop`             |
| `fresh_context` (`fresh`)   | Clear chat history before running (`true`/`false`)                | `true`             |

The picker pre-selects the scenario's `safety` policy and `fresh_context`, but
you can override both in the **Safety & options** panel before running.

## Safety policies

| Policy       | Behavior when a step looks like it failed                                                                 |
| ------------ | --------------------------------------------------------------------------------------------------------- |
| `stop`       | **Safe default.** A cheap heuristic tripwire halts the run and triggers an emergency stop.                |
| `supervise`  | An **AI safety supervisor** (LLM) adjudicates the failure and may confirm success, retry, mitigate with one corrective instruction, or abort. Aborts trigger an emergency stop. |
| `continue`   | Log issues but never auto-halt. Only the manual **EMERGENCY STOP** stops the robot.                       |

`supervise` requires an LLM-backed agent; in `--simple` mode (no LLM) it
automatically falls back to `stop`.

## Controls

- **▶ Run** — execute the (possibly edited) scenario in the editor box.
- **⏸ Pause / ▶ Resume** — pause takes effect *between* steps.
- **⏹ Stop** — stop the run and zero velocity.
- **🛑 EMERGENCY STOP** — cancel immediately and stop the robot (also on the Home/Status tabs).

## Bundled scenarios

| File                       | What it does                                                        | Needs                      |
| -------------------------- | ------------------------------------------------------------------ | -------------------------- |
| `welcome_demo.txt`         | Movement-free greeting + self-check. Great first run.              | nothing (works simulated)  |
| `system_diagnostics.txt`   | ROS graph + health sweep, no motion.                              | ROS 2 stack                |
| `garden_patrol.txt`        | 1 m square patrol via relative moves.                            | base driver + odometry     |
| `object_inspection.txt`    | Raise arm cam, capture, locate objects.                          | arm + MMC + detector       |
| `pick_and_deliver.txt`     | Localize → pick → hand over.                                     | arm + MMC skill server     |
| `return_home_and_park.txt` | Nav2 to map origin, then park the arm.                           | Nav2 + arm                 |

## Add your own

Drop a new `*.txt` file here following the format above, then click
**↻ Reload** in the Scenarios tab. Point the UI at a different folder with the
`SCENARIOS_DIR` environment variable.
