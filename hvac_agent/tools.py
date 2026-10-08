"""Tools the agent can call. Each returns small, JSON-serializable evidence.

The same Toolbox backs both the Claude agent and the offline rule-based agent,
so their results are directly comparable in the evaluation harness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .detect import compute_signals
from .simulate import DAYS, EVAL_DAYS, TRAIN_DAYS

ROOT_CAUSES = [
    "control_override",
    "setpoint_change",
    "efficiency_degradation",
    "sensor_fault",
    "data_gap",
    "operational_change",
    "weather_or_common_cause",
    "unexplained",
]

ROUTES = ["ship_engineering", "data_engineering", "no_action"]


@dataclass
class Diagnosis:
    unit_id: str
    root_cause: str
    confidence: float
    evidence: list[str]
    recommended_action: str
    route_to: str
    excess_kwh: float = 0.0
    onset: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), 3)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class Toolbox:
    def __init__(self, telemetry: pd.DataFrame, signals: pd.DataFrame | None = None):
        self.telemetry = telemetry
        self.signals = compute_signals(telemetry) if signals is None else signals
        self.diagnoses: dict[str, Diagnosis] = {}
        self.calls: list[tuple[str, dict]] = []  # audit trail

    # ------------------------------------------------------------------ tools
    def get_fleet_overview(self, ship_id: str | None = None, top_n: int = 15) -> dict:
        s = self.signals
        if ship_id:
            s = s[s["ship_id"] == ship_id]
        flagged = s[s["flagged"]].sort_values("excess_kwh", ascending=False)
        cols = ["unit_id", "ship_id", "excess_kwh", "excess_pct", "demand_z", "data_quality_issue"]
        return _clean({
            "units_reviewed": int(len(s)),
            "units_flagged": int(len(flagged)),
            "total_excess_kwh_flagged": float(flagged["excess_kwh"].clip(lower=0).sum()),
            "flagged": flagged[cols].head(top_n).to_dict("records"),
            "note": "Data-quality flags are listed even when excess energy looks normal or negative.",
        })

    def get_unit_diagnostics(self, unit_id: str) -> dict:
        r = self._row(unit_id)
        return _clean({
            "unit_id": unit_id,
            "energy": {k: r[k] for k in ["expected_kwh", "actual_kwh", "excess_kwh", "excess_pct", "demand_z", "onset"]},
            "behaviour": {
                "fan_speed_deviation_pct_points": r["speed_deviation_pct"],
                "fan_speed_deviation_day_pct_points": r["speed_deviation_day_pct"],
                "fan_speed_deviation_night_pct_points": r["speed_deviation_night_pct"],
                "share_hours_fan_near_max_pct": r["share_hours_near_max_speed"],
                "supply_setpoint_shift_c": r["setpoint_shift_c"],
                "efficiency_drift_pct": r["efficiency_drift_pct"],
                "efficiency_trend_pct_per_day": r["efficiency_trend_pct_per_day"],
            },
            "how_to_read": {
                "fan_speed_deviation": "fan running harder than its control logic normally would (>~10 pts is abnormal)",
                "day_vs_night": "deviation concentrated in normally-unoccupied night hours, with the fan still "
                                "following load, suggests the space is now in use (schedule change), not a fault",
                "near_max": "fan pinned near max most hours regardless of load suggests an override",
                "setpoint_shift": "negative = supply air setpoint lowered vs. baseline (>1 C is abnormal)",
                "efficiency_drift": "power above what this speed/load used to cost (>~5% abnormal; rising trend = degradation)",
            },
        })

    def check_data_quality(self, unit_id: str) -> dict:
        r = self._row(unit_id)
        return _clean({
            "unit_id": unit_id,
            "missing_pct_last_7d": r["missing_pct"],
            "longest_flatline_hours": r["flatline_hours"],
            "data_quality_issue": r["data_quality_issue"],
            "thresholds": {"missing_pct": 20, "flatline_hours": 12},
        })

    def compare_to_ship_peers(self, unit_id: str) -> dict:
        r = self._row(unit_id)
        peers = self.signals[(self.signals["ship_id"] == r["ship_id"]) & (self.signals["unit_id"] != unit_id)]
        pct_rank = float((peers["demand_z"] < r["demand_z"]).mean() * 100)
        return _clean({
            "unit_id": unit_id,
            "ship_id": r["ship_id"],
            "unit_demand_z": r["demand_z"],
            "peer_median_demand_z": float(peers["demand_z"].median()),
            "peer_share_flagged_pct": float(peers["flagged"].mean() * 100),
            "unit_percentile_vs_peers": pct_rank,
            "interpretation_hint": "If most peers deviate the same way, suspect weather or a ship-wide cause.",
        })

    def get_weather_context(self, ship_id: str) -> dict:
        tel = self.telemetry[self.telemetry["unit_id"] == self.signals[self.signals["ship_id"] == ship_id].index[0]]
        t0 = tel["timestamp"].min()
        train = tel[tel["timestamp"] < t0 + pd.Timedelta(days=TRAIN_DAYS)]["outside_temp_c"]
        ev = tel[tel["timestamp"] >= t0 + pd.Timedelta(days=DAYS - EVAL_DAYS)]["outside_temp_c"]
        return _clean({
            "ship_id": ship_id,
            "baseline_mean_temp_c": float(train.mean()),
            "last_7d_mean_temp_c": float(ev.mean()),
            "last_7d_max_temp_c": float(ev.max()),
            "baseline_p99_temp_c": float(train.quantile(0.99)),
        })

    def submit_diagnosis(self, unit_id: str, root_cause: str, confidence: float,
                         evidence: list[str], recommended_action: str, route_to: str) -> dict:
        if root_cause not in ROOT_CAUSES:
            return {"error": f"root_cause must be one of {ROOT_CAUSES}"}
        if route_to not in ROUTES:
            return {"error": f"route_to must be one of {ROUTES}"}
        r = self._row(unit_id)
        self.diagnoses[unit_id] = Diagnosis(
            unit_id=unit_id, root_cause=root_cause, confidence=float(confidence),
            evidence=list(evidence), recommended_action=recommended_action, route_to=route_to,
            excess_kwh=round(float(r["excess_kwh"]), 1), onset=r["onset"],
            meta={"ship_id": r["ship_id"]},
        )
        return {"status": "recorded", "unit_id": unit_id}

    # --------------------------------------------------------------- plumbing
    def _row(self, unit_id: str) -> pd.Series:
        if unit_id not in self.signals.index:
            raise KeyError(f"unknown unit_id {unit_id!r}")
        return self.signals.loc[unit_id]

    def call(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except (KeyError, TypeError, ValueError) as e:
            return {"error": str(e)}


TOOL_SCHEMAS = [
    {
        "name": "get_fleet_overview",
        "description": "List units flagged in the last 7 days, ranked by excess energy (kWh) versus "
                       "their weather-adjusted baseline. Start here.",
        "input_schema": {"type": "object", "properties": {
            "ship_id": {"type": "string", "description": "Optional, e.g. 'S1'."},
            "top_n": {"type": "integer", "default": 15}}},
    },
    {
        "name": "get_unit_diagnostics",
        "description": "Energy, fan-speed, setpoint and efficiency signals for one unit. Use these to "
                       "distinguish root causes.",
        "input_schema": {"type": "object", "properties": {"unit_id": {"type": "string"}},
                         "required": ["unit_id"]},
    },
    {
        "name": "check_data_quality",
        "description": "Missing-data share and longest frozen-value run for one unit. Always check before "
                       "trusting energy numbers.",
        "input_schema": {"type": "object", "properties": {"unit_id": {"type": "string"}},
                         "required": ["unit_id"]},
    },
    {
        "name": "compare_to_ship_peers",
        "description": "How this unit's deviation compares with the other units on the same ship.",
        "input_schema": {"type": "object", "properties": {"unit_id": {"type": "string"}},
                         "required": ["unit_id"]},
    },
    {
        "name": "get_weather_context",
        "description": "Recent outside temperature for a ship versus its baseline period.",
        "input_schema": {"type": "object", "properties": {"ship_id": {"type": "string"}},
                         "required": ["ship_id"]},
    },
    {
        "name": "submit_diagnosis",
        "description": "Record the final diagnosis for one unit. Call exactly once per investigated unit.",
        "input_schema": {"type": "object", "properties": {
            "unit_id": {"type": "string"},
            "root_cause": {"type": "string", "enum": ROOT_CAUSES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "array", "items": {"type": "string"},
                         "description": "Short, numeric facts from the tools that support the diagnosis."},
            "recommended_action": {"type": "string",
                                   "description": "Concrete next step for the receiving team."},
            "route_to": {"type": "string", "enum": ROUTES}},
            "required": ["unit_id", "root_cause", "confidence", "evidence", "recommended_action", "route_to"]},
    },
]
TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}
