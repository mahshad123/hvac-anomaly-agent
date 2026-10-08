"""Investigation agents.

Both agents follow the same diagnostic workflow an energy analyst uses:

    1. triage      which units are using more energy than they should?
    2. data check  can we trust this unit's data at all?
    3. model check is it the unit, or the weather / the whole ship?
    4. diagnose    which behaviour signal explains the excess?
    5. route       ship engineering (fix the equipment) or data engineering (fix the pipeline)

``RuleBasedAgent`` encodes that workflow as explicit rules. It runs offline, is
deterministic, and serves as a transparent baseline. ``ClaudeAgent`` gets the same
tools and the same workflow as instructions, and decides the steps itself through
tool calls.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .tools import ROOT_CAUSES, TOOL_SCHEMAS, Diagnosis, Toolbox

ACTIONS = {
    "control_override": "Check the BMS for a manual override or a VFD stuck in hand mode; return the fan to auto.",
    "setpoint_change": "Confirm whether the supply-air setpoint change was intentional; restore it to baseline if not.",
    "efficiency_degradation": "Inspect and clean the filters and cooling coil; check the belt and damper positions.",
    "sensor_fault": "Power meter is reporting a frozen value. Raise a ticket to verify the meter and connector.",
    "data_gap": "Most readings are missing. Raise a ticket to check the data connector or network link.",
    "operational_change": "No fault suspected. The fan follows load, but the space now runs at night. Confirm the new schedule with the hotel team.",
    "weather_or_common_cause": "No unit-level action. The deviation is shared across the ship.",
    "unexplained": "Ask the ship team for a walk-down inspection. No single signal explains the excess.",
}


class RuleBasedAgent:
    name = "rule_based"

    def run(self, toolbox: Toolbox, ship_id: str | None = None) -> list[Diagnosis]:
        overview = toolbox.call("get_fleet_overview", {"ship_id": ship_id, "top_n": 1000})
        for unit in overview["flagged"]:
            uid = unit["unit_id"]
            dq = toolbox.call("check_data_quality", {"unit_id": uid})
            if dq["data_quality_issue"]:
                if dq["longest_flatline_hours"] >= dq["thresholds"]["flatline_hours"]:
                    self._submit(toolbox, uid, "sensor_fault", 0.9,
                                 [f"power value frozen for {dq['longest_flatline_hours']} h"], "data_engineering")
                else:
                    self._submit(toolbox, uid, "data_gap", 0.9,
                                 [f"{dq['missing_pct_last_7d']:.0f}% of readings missing"], "data_engineering")
                continue

            peers = toolbox.call("compare_to_ship_peers", {"unit_id": uid})
            if peers["peer_share_flagged_pct"] > 50:
                self._submit(toolbox, uid, "weather_or_common_cause", 0.7,
                             [f"{peers['peer_share_flagged_pct']:.0f}% of ship peers also flagged"], "no_action")
                continue

            d = toolbox.call("get_unit_diagnostics", {"unit_id": uid})
            b, e = d["behaviour"], d["energy"]
            ev = [f"excess {e['excess_kwh']:.0f} kWh (+{e['excess_pct']:.0f}%) over 7 days"]
            night, day = b["fan_speed_deviation_night_pct_points"], b["fan_speed_deviation_day_pct_points"]
            if b["fan_speed_deviation_pct_points"] > 10 and b["share_hours_fan_near_max_pct"] > 50:
                cause = "control_override"
                ev.append(f"fan speed {b['fan_speed_deviation_pct_points']:+.0f} pts above expected, "
                          f"near max {b['share_hours_fan_near_max_pct']:.0f}% of hours")
            elif night > 10 and day < 5:
                ev.append(f"fan speed {night:+.0f} pts at night vs {day:+.0f} pts by day: looks like the space is now in use overnight")
                self._submit(toolbox, uid, "operational_change", 0.6, ev, "no_action")
                continue
            elif b["supply_setpoint_shift_c"] < -1:
                cause = "setpoint_change"
                ev.append(f"supply setpoint shifted {b['supply_setpoint_shift_c']:+.1f} C")
            elif b["efficiency_drift_pct"] > 5:
                cause = "efficiency_degradation"
                ev.append(f"power {b['efficiency_drift_pct']:+.0f}% above baseline for the same work, "
                          f"trend {b['efficiency_trend_pct_per_day']:+.1f}%/day")
            else:
                cause = "unexplained"
            self._submit(toolbox, uid, cause, 0.8 if cause != "unexplained" else 0.4, ev, "ship_engineering")
        return list(toolbox.diagnoses.values())

    @staticmethod
    def _submit(tb, uid, cause, conf, evidence, route):
        tb.call("submit_diagnosis", {"unit_id": uid, "root_cause": cause, "confidence": conf,
                                     "evidence": evidence, "recommended_action": ACTIONS[cause],
                                     "route_to": route})


SYSTEM_PROMPT = f"""You are an energy-efficiency analyst for a fleet of ships' HVAC air-handling units (AHUs).
Your job is to review the last 7 days, find units wasting energy or reporting bad data, work out why, and route each
issue to the team that can fix it. Ship crews are busy, so do not page them about weather or bad data.

Workflow. Follow it for every flagged unit:
1. Call get_fleet_overview to triage.
2. For each flagged unit, call check_data_quality first. Frozen values point to sensor_fault, and heavy missingness
   points to data_gap. Route both to data_engineering, and do not interpret their energy numbers.
3. Call compare_to_ship_peers. If most of the ship deviates together, check get_weather_context and consider
   weather_or_common_cause (route no_action).
4. Call get_unit_diagnostics. Use the behaviour signals to pick the root cause:
   a fan pinned near max regardless of load suggests control_override; extra fan speed only during normally
   unoccupied night hours, with the fan still following load, suggests operational_change (a legitimate schedule
   change: route no_action, and say the schedule should be confirmed); a setpoint shifted down suggests setpoint_change;
   more power for the same work, rising over time, suggests efficiency_degradation; anything else is unexplained.
5. Call submit_diagnosis exactly once per flagged unit. Evidence must be short, numeric, and quoted from tool
   output. Never invent numbers.

Allowed root causes: {", ".join(ROOT_CAUSES)}.
When every flagged unit has a diagnosis, reply with a one-paragraph summary and stop."""


class ClaudeAgent:
    """Tool-using LLM agent built on the Anthropic Messages API."""

    name = "claude"

    def __init__(self, model: str | None = None, client: Any = None, max_turns: int = 80):
        self.model = model or os.environ.get("HVAC_AGENT_MODEL", "claude-sonnet-4-5")
        self.max_turns = max_turns
        if client is None:
            import anthropic  # imported lazily so the offline path has no dependency

            client = anthropic.Anthropic()
        self.client = client
        self.summary: str = ""

    def run(self, toolbox: Toolbox, ship_id: str | None = None) -> list[Diagnosis]:
        scope = f"ship {ship_id}" if ship_id else "the whole fleet"
        messages: list[dict] = [{"role": "user", "content": f"Review {scope} for the last 7 days."}]
        for _ in range(self.max_turns):
            resp = self.client.messages.create(
                model=self.model, max_tokens=2048, system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS, messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                self.summary = "".join(getattr(b, "text", "") for b in resp.content)
                break
            results = []
            for block in tool_uses:
                out = toolbox.call(block.name, dict(block.input))
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(out)})
            messages.append({"role": "user", "content": results})
        return list(toolbox.diagnoses.values())
