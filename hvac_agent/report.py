"""Render diagnoses as messages that a ship team or data team can act on."""

from __future__ import annotations

from collections import defaultdict

from .tools import Diagnosis

LABEL = {
    "control_override": "Fan running at full speed (possible manual override)",
    "setpoint_change": "Supply-air setpoint lowered",
    "efficiency_degradation": "Efficiency degrading (possible fouled coil or filter)",
    "sensor_fault": "Power meter frozen",
    "data_gap": "Telemetry missing",
    "operational_change": "Schedule change, no fault",
    "weather_or_common_cause": "Ship-wide or weather-driven, no action",
    "unexplained": "Excess energy, cause unclear",
}


def render_report(diagnoses: list[Diagnosis], kwh_price: float = 0.15) -> str:
    by_route = defaultdict(list)
    for d in diagnoses:
        by_route[d.route_to].append(d)

    lines = ["# Weekly HVAC energy review", ""]
    ship = sorted(by_route["ship_engineering"], key=lambda d: -d.excess_kwh)
    total = sum(max(d.excess_kwh, 0) for d in ship)
    lines += [
        f"**{len(ship)} units need attention from ship engineering.** Together they used about "
        f"**{total:,.0f} kWh** more than expected last week (about ${total * kwh_price:,.0f} at ${kwh_price}/kWh).",
        f"{len(by_route['data_engineering'])} units have data problems and went to data engineering. "
        f"{len(by_route['no_action'])} deviations were explained by schedule changes or ship-wide conditions and need no repair.",
        "",
    ]

    if ship:
        lines += ["## For ship engineering", "",
                  "| Unit | Issue | Excess kWh (7d) | Since | Confidence |", "|---|---|---|---|---|"]
        for d in ship:
            lines.append(f"| {d.unit_id} | {LABEL[d.root_cause]} | {d.excess_kwh:,.0f} | "
                         f"{d.onset or 'n/a'} | {d.confidence:.0%} |")
        lines.append("")
        for d in ship:
            lines += [f"**{d.unit_id}: {LABEL[d.root_cause]}**", "",
                      *[f"- {e}" for e in d.evidence], f"- **Action:** {d.recommended_action}", ""]

    if by_route["data_engineering"]:
        lines += ["## For data engineering", ""]
        for d in sorted(by_route["data_engineering"], key=lambda d: d.unit_id):
            lines.append(f"- **{d.unit_id}** ({LABEL[d.root_cause].lower()}, {'; '.join(d.evidence)}). "
                         f"{d.recommended_action.split('. ', 1)[-1]}")
        lines.append("")

    if by_route["no_action"]:
        lines += ["## Reviewed, no action", ""]
        for d in sorted(by_route["no_action"], key=lambda d: d.unit_id):
            lines.append(f"- {d.unit_id}: {'; '.join(d.evidence)}")
    return "\n".join(lines).rstrip() + "\n"
