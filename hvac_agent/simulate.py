"""Synthetic fleet of air-handling units (AHUs) with labelled, injected faults.

Everything here is simulated, so the project is fully reproducible and no
proprietary data is involved. The physics is simplified but keeps the
relationships that matter for diagnosis:

    fan power      ~ speed^3                    (fan affinity laws)
    cooling power  ~ speed * (T_outside - supply setpoint)
    fan speed      ~ occupancy schedule + outside temperature

Fault types (the root-cause taxonomy the agent must recover):

    control_override        fan VFD stuck at 100% (manual override left on)
    setpoint_change         supply-air setpoint lowered by 3-4 °C
    efficiency_degradation  coil/filter fouling: same work costs 0 -> +40% more power
    sensor_fault            power meter frozen at a constant value
    data_gap                most readings missing (connector / network issue)

Plus two kinds of *non-fault* that look like faults to a naive detector:

    heatwave                one ship's outside temperature jumps +5 °C for four days
    operational_change      a space moves to 24h operation (e.g. an event venue), so its
                            fan legitimately runs harder at night. Labelled "none".

Magnitudes are randomised so some faults are subtle, and every unit has a few
percent of readings randomly missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FAULT_TYPES = [
    "control_override",
    "setpoint_change",
    "efficiency_degradation",
    "sensor_fault",
    "data_gap",
]

DAYS = 42
TRAIN_DAYS = 21  # first three weeks are fault-free and used to fit baselines
EVAL_DAYS = 7  # the agent reviews the most recent week


@dataclass
class FleetConfig:
    n_ships: int = 3
    ahus_per_ship: int = 40
    faults_per_type: int = 3  # across the fleet
    operational_changes: int = 3  # legitimate schedule changes (not faults)
    heatwave_ship: int | None = 1  # index of ship that hits a heatwave (None = off)
    seed: int = 7


def _weather(rng, hours: np.ndarray, heatwave: bool) -> tuple[np.ndarray, np.ndarray]:
    day = hours / 24
    temp = (
        27.0
        + 3.5 * np.sin(2 * np.pi * (hours % 24 - 9) / 24)  # diurnal cycle, peak ~15:00
        + 1.5 * np.sin(2 * np.pi * day / 11)  # weather systems / itinerary
        + rng.normal(0, 0.6, len(hours))
    )
    if heatwave:
        start, end = (DAYS - 6) * 24, (DAYS - 2) * 24
        temp[start:end] += 5.0
    humidity = np.clip(70 - 1.2 * (temp - 27) + rng.normal(0, 4, len(hours)), 35, 98)
    return temp, humidity


def simulate_fleet(cfg: FleetConfig = FleetConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (telemetry, ground_truth)."""
    rng = np.random.default_rng(cfg.seed)
    n_hours = DAYS * 24
    hours = np.arange(n_hours)
    ts = pd.Timestamp("2026-08-01") + pd.to_timedelta(hours, unit="h")
    hod = hours % 24
    occupancy = np.where((hod >= 7) & (hod <= 22), 1.0, 0.35)

    # Pick faulty units (distinct) across the fleet.
    unit_ids = [f"S{s + 1}-AHU{u + 1:03d}" for s in range(cfg.n_ships) for u in range(cfg.ahus_per_ship)]
    n_faulty = cfg.faults_per_type * len(FAULT_TYPES)
    chosen = rng.choice(len(unit_ids), size=n_faulty + cfg.operational_changes, replace=False)
    fault_of = {}
    for i, idx in enumerate(chosen):
        ftype = FAULT_TYPES[i % len(FAULT_TYPES)] if i < n_faulty else "operational_change"
        start_day = int(rng.integers(TRAIN_DAYS + 2, DAYS - EVAL_DAYS + 1))
        fault_of[unit_ids[idx]] = (ftype, start_day * 24)

    frames, truth = [], []
    for s in range(cfg.n_ships):
        ship = f"S{s + 1}"
        temp, hum = _weather(rng, hours, heatwave=(cfg.heatwave_ship == s))
        for u in range(cfg.ahus_per_ship):
            uid = f"{ship}-AHU{u + 1:03d}"
            fan_kw = rng.uniform(4, 18)
            k_cool = rng.uniform(0.25, 0.6) * fan_kw / 10
            base_sp = rng.choice([12.0, 13.0, 14.0])
            base_kw = rng.uniform(0.3, 1.0)

            ftype, start = fault_of.get(uid, (None, None))
            occ = occupancy.copy()
            if ftype == "operational_change":
                occ[start:] = 1.0  # space now in use around the clock
            speed = np.clip(
                30 + 35 * occ + 3.0 * (temp - 27) + rng.normal(0, 3, n_hours), 20, 100
            )
            setpoint = np.full(n_hours, base_sp) + rng.normal(0, 0.05, n_hours)
            efficiency = np.ones(n_hours)

            if ftype == "control_override":
                speed[start:] = rng.uniform(85, 100) + rng.normal(0, 0.3, n_hours - start)
            elif ftype == "setpoint_change":
                setpoint[start:] -= rng.uniform(1.5, 4.0)
            elif ftype == "efficiency_degradation":
                ramp = np.clip((hours[start:] - start) / (14 * 24), 0, 1)
                efficiency[start:] = 1 + rng.uniform(0.15, 0.4) * ramp

            s_frac = speed / 100
            power = efficiency * (
                base_kw + fan_kw * s_frac**3 + k_cool * s_frac * np.maximum(temp - setpoint, 0)
            ) * (1 + rng.normal(0, 0.04, n_hours))
            # Background packet loss on every unit.
            drop = rng.random(n_hours) < 0.02
            power[drop] = np.nan

            if ftype == "sensor_fault":
                power[start:] = round(float(np.nanmean(power[start - 6:start])), 3)
            if ftype == "data_gap":
                mask = rng.random(n_hours - start) < 0.7
                power[start:][mask] = np.nan
                speed[start:][mask] = np.nan

            frames.append(pd.DataFrame({
                "ship_id": ship, "unit_id": uid, "timestamp": ts,
                "outside_temp_c": temp.round(2), "humidity_pct": hum.round(1),
                "supply_setpoint_c": setpoint.round(2), "fan_speed_pct": speed.round(1),
                "power_kw": power.round(3),
            }))
            is_fault = ftype in FAULT_TYPES
            truth.append({"ship_id": ship, "unit_id": uid, "fault": ftype if is_fault else "none",
                          "scenario": ftype or "normal",
                          "start_hour": start, "start_time": ts[start] if start is not None else None})

    return pd.concat(frames, ignore_index=True), pd.DataFrame(truth)
