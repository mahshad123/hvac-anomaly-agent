from types import SimpleNamespace

import numpy as np
import pytest

from hvac_agent.agent import ClaudeAgent, RuleBasedAgent
from hvac_agent.detect import naive_threshold_flags
from hvac_agent.evaluate import score, score_naive
from hvac_agent.report import render_report
from hvac_agent.simulate import FAULT_TYPES, FleetConfig, simulate_fleet
from hvac_agent.tools import ROOT_CAUSES, TOOL_NAMES, Toolbox


@pytest.fixture(scope="module")
def fleet():
    tel, truth = simulate_fleet(FleetConfig(seed=11))
    return tel, truth, Toolbox(tel).signals


def test_simulator_is_deterministic():
    a, ta = simulate_fleet(FleetConfig(n_ships=1, ahus_per_ship=10, faults_per_type=1, operational_changes=1, seed=3))
    b, tb = simulate_fleet(FleetConfig(n_ships=1, ahus_per_ship=10, faults_per_type=1, operational_changes=1, seed=3))
    assert a.equals(b) and ta.equals(tb)


def test_every_fault_type_injected(fleet):
    _, truth, _ = fleet
    assert set(FAULT_TYPES) <= set(truth["fault"])


def test_healthy_units_mostly_not_flagged(fleet):
    _, truth, sig = fleet
    normal = truth.loc[truth["scenario"] == "normal", "unit_id"]
    assert sig.loc[normal, "flagged"].mean() < 0.03


def test_signals_separate_root_causes(fleet):
    _, truth, sig = fleet
    t = truth.set_index("unit_id")
    def units(f): return t.index[t["fault"] == f]
    assert (sig.loc[units("control_override"), "share_hours_near_max_speed"] > 50).all()
    assert (sig.loc[units("setpoint_change"), "setpoint_shift_c"] < -1).all()
    assert (sig.loc[units("sensor_fault"), "flatline_hours"] >= 12).all()
    assert (sig.loc[units("data_gap"), "missing_pct"] > 20).all()


def test_rule_agent_beats_naive_threshold(fleet):
    tel, truth, sig = fleet
    agent = score(RuleBasedAgent().run(Toolbox(tel, sig)), truth)
    naive = score_naive(naive_threshold_flags(tel), truth)
    assert agent["detect_f1"] > naive["detect_f1"] + 0.3
    assert agent["false_crew_pages"] < naive["false_crew_pages"]
    assert agent["routing_accuracy"] >= 0.9


def test_data_faults_go_to_data_engineering(fleet):
    tel, truth, sig = fleet
    diags = {d.unit_id: d for d in RuleBasedAgent().run(Toolbox(tel, sig))}
    for uid in truth.loc[truth["fault"].isin(["sensor_fault", "data_gap"]), "unit_id"]:
        assert diags[uid].route_to == "data_engineering"


def test_toolbox_rejects_bad_inputs(fleet):
    tel, _, sig = fleet
    tb = Toolbox(tel, sig)
    assert "error" in tb.call("get_unit_diagnostics", {"unit_id": "NOPE"})
    assert "error" in tb.call("delete_everything", {})
    assert "error" in tb.call("submit_diagnosis", {
        "unit_id": sig.index[0], "root_cause": "aliens", "confidence": 1,
        "evidence": [], "recommended_action": "", "route_to": "no_action"})


def test_tool_outputs_are_json_safe(fleet):
    import json
    tel, _, sig = fleet
    tb = Toolbox(tel, sig)
    uid = sig.index[0]
    for name, args in [("get_fleet_overview", {}), ("get_unit_diagnostics", {"unit_id": uid}),
                       ("check_data_quality", {"unit_id": uid}), ("compare_to_ship_peers", {"unit_id": uid}),
                       ("get_weather_context", {"ship_id": "S1"})]:
        json.dumps(tb.call(name, args), allow_nan=False)


def test_report_renders(fleet):
    tel, _, sig = fleet
    md = render_report(RuleBasedAgent().run(Toolbox(tel, sig)))
    assert "For ship engineering" in md and "For data engineering" in md


# ---------------- Claude agent loop, with a scripted fake client ----------------
class FakeClient:
    """Plays back a fixed sequence of assistant turns, recording what it was sent."""

    def __init__(self, unit_id):
        self.sent = []
        self.turns = [
            [("tool_use", "get_fleet_overview", {})],
            [("tool_use", "check_data_quality", {"unit_id": unit_id}),
             ("tool_use", "get_unit_diagnostics", {"unit_id": unit_id})],
            [("tool_use", "submit_diagnosis", {
                "unit_id": unit_id, "root_cause": "control_override", "confidence": 0.9,
                "evidence": ["fan near max 100% of hours"], "recommended_action": "check BMS",
                "route_to": "ship_engineering"})],
            [("text", "Done. One unit diagnosed.")],
        ]
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.sent.append(kw)
        blocks = []
        for i, b in enumerate(self.turns.pop(0)):
            if b[0] == "tool_use":
                blocks.append(SimpleNamespace(type="tool_use", id=f"t{i}", name=b[1], input=b[2]))
            else:
                blocks.append(SimpleNamespace(type="text", text=b[1]))
        stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
        return SimpleNamespace(content=blocks, stop_reason=stop)


def test_claude_agent_tool_loop(fleet):
    tel, truth, sig = fleet
    uid = truth.loc[truth["fault"] == "control_override", "unit_id"].iat[0]
    client = FakeClient(uid)
    tb = Toolbox(tel, sig)
    diags = ClaudeAgent(client=client, model="test-model").run(tb)

    assert [d.unit_id for d in diags] == [uid]
    assert diags[0].root_cause == "control_override"
    assert [c[0] for c in tb.calls] == ["get_fleet_overview", "check_data_quality",
                                        "get_unit_diagnostics", "submit_diagnosis"]
    # Tool results are returned to the model with matching ids.
    # Tool results go back to the model with ids matching the tool_use blocks.
    msgs = client.sent[-1]["messages"]
    results = [m for m in msgs if m["role"] == "user" and isinstance(m["content"], list)]
    assert [len(m["content"]) for m in results] == [1, 2, 1]
    assert [r["tool_use_id"] for r in results[1]["content"]] == ["t0", "t1"]
    assert {t["name"] for t in client.sent[0]["tools"]} == TOOL_NAMES


def test_schema_enum_matches_taxonomy():
    from hvac_agent.tools import TOOL_SCHEMAS
    submit = next(t for t in TOOL_SCHEMAS if t["name"] == "submit_diagnosis")
    assert submit["input_schema"]["properties"]["root_cause"]["enum"] == ROOT_CAUSES
