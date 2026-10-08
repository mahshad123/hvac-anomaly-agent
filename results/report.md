# Weekly HVAC energy review

**9 units need attention from ship engineering.** Together they used about **4,893 kWh** more than expected last week (about $734 at $0.15/kWh).
6 units have data problems and went to data engineering. 3 deviations were explained by schedule changes or ship-wide conditions and need no repair.

## For ship engineering

| Unit | Issue | Excess kWh (7d) | Since | Confidence |
|---|---|---|---|---|
| S3-AHU018 | Fan running at full speed (possible manual override) | 1,492 | 2026-09-03 | 80% |
| S3-AHU011 | Fan running at full speed (possible manual override) | 1,353 | 2026-08-30 | 80% |
| S3-AHU040 | Fan running at full speed (possible manual override) | 1,213 | 2026-08-31 | 80% |
| S3-AHU016 | Efficiency degrading (possible fouled coil or filter) | 351 | 2026-08-26 | 80% |
| S2-AHU026 | Supply-air setpoint lowered | 196 | 2026-09-02 | 80% |
| S1-AHU007 | Supply-air setpoint lowered | 101 | 2026-08-25 | 80% |
| S1-AHU016 | Supply-air setpoint lowered | 73 | 2026-08-26 | 80% |
| S3-AHU025 | Efficiency degrading (possible fouled coil or filter) | 68 | 2026-09-05 | 80% |
| S3-AHU004 | Efficiency degrading (possible fouled coil or filter) | 46 | 2026-09-04 | 80% |

**S3-AHU018: Fan running at full speed (possible manual override)**

- excess 1492 kWh (+132%) over 7 days
- fan speed +38 pts above expected, near max 100% of hours
- **Action:** Check the BMS for a manual override or a VFD stuck in hand mode; return the fan to auto.

**S3-AHU011: Fan running at full speed (possible manual override)**

- excess 1353 kWh (+116%) over 7 days
- fan speed +35 pts above expected, near max 100% of hours
- **Action:** Check the BMS for a manual override or a VFD stuck in hand mode; return the fan to auto.

**S3-AHU040: Fan running at full speed (possible manual override)**

- excess 1213 kWh (+89%) over 7 days
- fan speed +30 pts above expected, near max 100% of hours
- **Action:** Check the BMS for a manual override or a VFD stuck in hand mode; return the fan to auto.

**S3-AHU016: Efficiency degrading (possible fouled coil or filter)**

- excess 351 kWh (+26%) over 7 days
- power +24% above baseline for the same work, trend +1.5%/day
- **Action:** Inspect and clean the filters and cooling coil; check the belt and damper positions.

**S2-AHU026: Supply-air setpoint lowered**

- excess 196 kWh (+9%) over 7 days
- supply setpoint shifted -2.9 C
- **Action:** Confirm whether the supply-air setpoint change was intentional; restore it to baseline if not.

**S1-AHU007: Supply-air setpoint lowered**

- excess 101 kWh (+12%) over 7 days
- supply setpoint shifted -3.2 C
- **Action:** Confirm whether the supply-air setpoint change was intentional; restore it to baseline if not.

**S1-AHU016: Supply-air setpoint lowered**

- excess 73 kWh (+8%) over 7 days
- supply setpoint shifted -2.7 C
- **Action:** Confirm whether the supply-air setpoint change was intentional; restore it to baseline if not.

**S3-AHU025: Efficiency degrading (possible fouled coil or filter)**

- excess 68 kWh (+8%) over 7 days
- power +9% above baseline for the same work, trend +0.6%/day
- **Action:** Inspect and clean the filters and cooling coil; check the belt and damper positions.

**S3-AHU004: Efficiency degrading (possible fouled coil or filter)**

- excess 47 kWh (+10%) over 7 days
- power +11% above baseline for the same work, trend +0.7%/day
- **Action:** Inspect and clean the filters and cooling coil; check the belt and damper positions.

## For data engineering

- **S1-AHU033** (telemetry missing, 69% of readings missing). Raise a ticket to check the data connector or network link.
- **S1-AHU034** (telemetry missing, 73% of readings missing). Raise a ticket to check the data connector or network link.
- **S2-AHU019** (power meter frozen, power value frozen for 168 h). Raise a ticket to verify the meter and connector.
- **S2-AHU022** (telemetry missing, 67% of readings missing). Raise a ticket to check the data connector or network link.
- **S3-AHU017** (power meter frozen, power value frozen for 168 h). Raise a ticket to verify the meter and connector.
- **S3-AHU020** (power meter frozen, power value frozen for 168 h). Raise a ticket to verify the meter and connector.

## Reviewed, no action

- S1-AHU001: excess 154 kWh (+12%) over 7 days; fan speed +22 pts at night vs -0 pts by day: looks like the space is now in use overnight
- S1-AHU025: excess 148 kWh (+12%) over 7 days; fan speed +23 pts at night vs +0 pts by day: looks like the space is now in use overnight
- S2-AHU032: excess 313 kWh (+13%) over 7 days; fan speed +24 pts at night vs +0 pts by day: looks like the space is now in use overnight
