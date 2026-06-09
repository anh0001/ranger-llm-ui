"""Tests for the LookAt tool (aim the wrist camera at a 3D point).

Run in simulation mode (no ROS 2 / hardware): the manipulation interface reports
simulation_mode and the tool returns a canned result, so these check the tool
surface (params, schema, registration) without a skill server.
"""

import pytest

from ranger_llm_ui.tools.manipulation_tools import (
    LookAtTool,
    LookAtInput,
    get_manipulation_tools,
)
from ranger_llm_ui.tools.all_tools import get_all_tools, get_tools_by_category


class TestLookAtTool:
    def setup_method(self):
        self.tool = LookAtTool()

    def test_identity(self):
        assert self.tool.name == "LookAt"
        assert self.tool.skill_name == "look_at"
        assert self.tool.args_schema is LookAtInput

    def test_args_schema_fields(self):
        fields = LookAtInput.model_fields
        assert set(fields) == {"x", "y", "z", "frame", "standoff_m"}
        assert fields["frame"].default == "base_footprint"
        assert fields["standoff_m"].default == 0.0

    def test_simulation_orientation_only(self):
        out = self.tool._run(x=0.4, y=-0.05, z=0.15)
        assert "simulated" in out.lower()
        # params passed to the skill are correct
        assert "'position': [0.4, -0.05, 0.15]" in out
        assert "'frame': 'base_footprint'" in out
        assert "'standoff_m': 0.0" in out

    def test_simulation_with_standoff_and_frame(self):
        out = self.tool._run(x=1.0, y=2.0, z=0.3, frame="map", standoff_m=0.25)
        assert "simulated" in out.lower()
        assert "'frame': 'map'" in out
        assert "'standoff_m': 0.25" in out

    def test_run_coerces_to_float(self):
        # ints in -> floats in the dispatched params (position is numeric)
        out = self.tool._run(x=0, y=0, z=0, standoff_m=0)
        assert "'position': [0.0, 0.0, 0.0]" in out


class TestLookAtRegistration:
    def test_in_manipulation_tools(self):
        # LookAt actuates the arm, so it is gated with the manipulation tools
        # (ENABLE_MANIPULATION_TOOLS) rather than the read-only perception tools.
        names = [t.name for t in get_manipulation_tools()]
        assert "LookAt" in names

    def test_in_get_all_tools(self):
        names = [t.name for t in get_all_tools()]
        assert "LookAt" in names
        assert "LocateObject" in names

    def test_in_manipulation_category(self):
        names = [t.name for t in get_tools_by_category("manipulation")]
        assert "LookAt" in names

    def test_disabled_when_manipulation_off(self):
        names = [t.name for t in get_all_tools(include_manipulation=False)]
        assert "LookAt" not in names


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
