"""Statistical layer: per-unit baselines, residuals, data-quality checks.

The LLM never looks at raw time series. It works from compact, verifiable
signals computed here, which keeps it cheap, auditable and hard to fool.

Three baselines are fitted per unit on the fault-free training window:

* demand model      power ~ weather + time-of-day      -> "is this unit using more energy
                                                          than it normally would today?"
* speed model       speed ~ weather + time-of-day      -> "is the fan running harder than its
                                                          control logic normally would?"
* efficiency model  power ~ speed^3 + speed*(T - setpoint) -> "given what it's doing, is it
                                                          paying more power than it used to?"

The pattern of residuals across the three models is what separates root causes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .simulate import DAYS, EVAL_DAYS, TRAIN_DAYS


def _context_features(df: pd.DataFrame) -> np.ndarray:
    t = df["outside_temp_c"].to_numpy()
    h = df["humidity_pct"].to_numpy()
    hod = df["timestamp"].dt.hour.to_numpy()
    occ = ((hod >= 7) & (hod <= 22)).astype(float)
    return np.column_stack([
        np.ones(len(df)), t, t**2, t**3, h, occ, occ * t,
        np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
    ])


def _efficiency_features(df: pd.DataFrame) -> np.ndarray:
    s = df["fan_speed_pct"].to_numpy() / 100
    lift = np.maximum(df["outside_temp_c"].to_numpy() - df["supply_setpoint_c"].to_numpy(), 0)
    return np.column_stack([np.ones(len(df)), s, s**3, s * lift])


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    ok = np.isfinite(y) & np.isfinite(x).all(1)
    coef, *_ = np.linalg.lstsq(x[ok], y[ok], rcond=None)
    resid = y[ok] - x[ok] @ coef
    # Robust scale (MAD) so a few spikes in training don't inflate thresholds.
    sd = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    return coef, max(float(sd), 1e-6)


def _predict(coef, x):
    return x @ coef


@dataclass
class UnitSignals:
    unit_id: str
    ship_id: str
    # energy
    expected_kwh: float
    actual_kwh: float
    excess_kwh: float
    excess_pct: float
    demand_z: float  # mean standardized residual of the demand model over the eval week
    onset: str | None  # CUSUM change-point estimate
    # behaviour
    speed_deviation_pct: float  # actual - expected fan speed, percentage points
    speed_deviation_day_pct: float  # same, 07:00-22:00 only
    speed_deviation_night_pct: float  # same, 23:00-06:00 only
    share_hours_near_max_speed: float  # % of eval hours with speed >= 85%
    setpoint_shift_c: float  # eval median - training median
    efficiency_drift_pct: float  # power vs efficiency model, % above baseline
    efficiency_trend_pct_per_day: float
    # data quality
    missing_pct: float
    flatline_hours: int
    # derived
    flagged: bool
    data_quality_issue: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def _longest_flat_run(x: np.ndarray, tol: float = 1e-9) -> int:
    best = cur = 0
    for a, b in zip(x[:-1], x[1:]):
        if np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best + 1 if best else 0


def _cusum_onset(daily_z: pd.Series, k: float = 0.5, h: float = 4.0) -> str | None:
    s = 0.0
    start = None
    for day, z in daily_z.items():
        if not np.isfinite(z):
            continue
        s_new = max(0.0, s + z - k)
        if s == 0.0 and s_new > 0:
            start = day
        s = s_new
        if s > h:
            return str(start.date())
        if s == 0.0:
            start = None
    return None


# Thresholds were calibrated once on simulator seed 1 (healthy units stay below
# |z| 0.75 and 3.6% excess) and then frozen. Evaluation uses different seeds.
Z_FLAG = 1.2
EXCESS_PCT_FLAG = 6.0
SETPOINT_FLAG_C = 1.0
MISSING_FLAG = 20.0
FLATLINE_FLAG = 12


def compute_signals(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Return one row of UnitSignals per unit."""
    t0 = telemetry["timestamp"].min()
    train_end = t0 + pd.Timedelta(days=TRAIN_DAYS)
    eval_start = t0 + pd.Timedelta(days=DAYS - EVAL_DAYS)

    rows = []
    for uid, df in telemetry.groupby("unit_id", sort=True):
        df = df.sort_values("timestamp")
        train = df[df["timestamp"] < train_end]
        post = df[df["timestamp"] >= train_end]
        ev = df[df["timestamp"] >= eval_start]

        xc_tr, xc_ev, xc_post = map(_context_features, (train, ev, post))
        demand_coef, demand_sd = _fit(xc_tr, train["power_kw"].to_numpy())
        speed_coef, _ = _fit(xc_tr, train["fan_speed_pct"].to_numpy())
        eff_coef, _ = _fit(_efficiency_features(train), train["power_kw"].to_numpy())

        p_ev = ev["power_kw"].to_numpy()
        ok = np.isfinite(p_ev)
        exp_ev = _predict(demand_coef, xc_ev)
        resid = p_ev[ok] - exp_ev[ok]
        expected = float(exp_ev[ok].sum()) if ok.any() else 0.0
        actual = float(p_ev[ok].sum()) if ok.any() else 0.0
        excess = actual - expected
        demand_z = float(resid.mean() / demand_sd) if ok.any() else float("nan")

        # Onset: daily mean z from end of training onwards.
        r_post = (post["power_kw"].to_numpy() - _predict(demand_coef, xc_post)) / demand_sd
        daily = pd.Series(r_post, index=post["timestamp"]).resample("D").mean()
        onset = _cusum_onset(daily)

        s_ev = ev["fan_speed_pct"].to_numpy()
        s_ok = np.isfinite(s_ev)
        s_res = s_ev - _predict(speed_coef, xc_ev)
        hod_ev = ev["timestamp"].dt.hour.to_numpy()
        day_mask = (hod_ev >= 7) & (hod_ev <= 22)

        def _m(mask):
            m = mask & s_ok
            return float(np.mean(s_res[m])) if m.any() else float("nan")

        speed_dev, speed_day, speed_night = _m(np.ones_like(s_ok)), _m(day_mask), _m(~day_mask)
        near_max = float(100 * np.mean(s_ev[s_ok] >= 85)) if s_ok.any() else float("nan")
        sp_shift = float(ev["supply_setpoint_c"].median() - train["supply_setpoint_c"].median())

        eff_pred_post = _predict(eff_coef, _efficiency_features(post))
        eff_ratio = post["power_kw"].to_numpy() / np.maximum(eff_pred_post, 1e-6)
        eff_daily = pd.Series(eff_ratio, index=post["timestamp"]).resample("D").mean().dropna()
        eff_ev = eff_ratio[post["timestamp"].to_numpy() >= np.datetime64(eval_start)]
        eff_drift = float((np.nanmean(eff_ev) - 1) * 100) if np.isfinite(eff_ev).any() else float("nan")
        eff_trend = float(np.polyfit(np.arange(len(eff_daily)), eff_daily.to_numpy(), 1)[0] * 100) \
            if len(eff_daily) > 2 else float("nan")

        missing = float(100 * (1 - ok.mean()))
        flat = _longest_flat_run(p_ev)
        # Background packet loss is normal; only sustained gaps matter.
        dq = missing > MISSING_FLAG or flat >= FLATLINE_FLAG
        excess_pct = 100 * excess / expected if expected > 0 else float("nan")
        config_change = abs(sp_shift) > SETPOINT_FLAG_C  # a direct signal, even if energy impact is small
        flagged = dq or config_change or (demand_z > Z_FLAG and excess_pct > EXCESS_PCT_FLAG)

        rows.append(UnitSignals(
            unit_id=uid, ship_id=df["ship_id"].iat[0],
            expected_kwh=expected, actual_kwh=actual, excess_kwh=excess, excess_pct=excess_pct,
            demand_z=demand_z, onset=onset, speed_deviation_pct=speed_dev,
            speed_deviation_day_pct=speed_day, speed_deviation_night_pct=speed_night,
            share_hours_near_max_speed=near_max,
            setpoint_shift_c=sp_shift, efficiency_drift_pct=eff_drift,
            efficiency_trend_pct_per_day=eff_trend, missing_pct=missing, flatline_hours=flat,
            flagged=bool(flagged), data_quality_issue=bool(dq),
        ).to_dict())
    return pd.DataFrame(rows).set_index("unit_id", drop=False)


def naive_threshold_flags(telemetry: pd.DataFrame) -> pd.Series:
    """Baseline most teams start with: flag if eval-week mean power > training P95 of daily means."""
    t0 = telemetry["timestamp"].min()
    out = {}
    for uid, df in telemetry.groupby("unit_id"):
        daily = df.set_index("timestamp")["power_kw"].resample("D").mean()
        train = daily[daily.index < t0 + pd.Timedelta(days=TRAIN_DAYS)]
        ev = daily[daily.index >= t0 + pd.Timedelta(days=DAYS - EVAL_DAYS)]
        out[uid] = bool(ev.mean() > train.quantile(0.95))
    return pd.Series(out)
