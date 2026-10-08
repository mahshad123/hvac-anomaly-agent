"""Score agent output against the simulator's ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tools import Diagnosis

DATA_FAULTS = {"sensor_fault", "data_gap"}


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def score(diagnoses: list[Diagnosis], truth: pd.DataFrame) -> dict[str, float]:
    truth = truth.set_index("unit_id")
    positives = set(truth.index[truth["fault"] != "none"])
    actioned = {d.unit_id: d for d in diagnoses if d.route_to != "no_action"}

    tp = len(positives & actioned.keys())
    fp = len(actioned.keys() - positives)
    fn = len(positives - actioned.keys())
    out = {f"detect_{k}": v for k, v in _prf(tp, fp, fn).items()}

    hits = [actioned[u] for u in positives & actioned.keys()]
    out["root_cause_accuracy"] = float(np.mean([d.root_cause == truth.at[d.unit_id, "fault"] for d in hits])) if hits else 0.0

    def right_team(d):
        expected = "data_engineering" if truth.at[d.unit_id, "fault"] in DATA_FAULTS else "ship_engineering"
        return d.route_to == expected
    out["routing_accuracy"] = float(np.mean([right_team(d) for d in hits])) if hits else 0.0

    # Crew pages that were not real equipment problems = wasted crew time.
    crew = [d for d in actioned.values() if d.route_to == "ship_engineering"]
    out["false_crew_pages"] = sum(truth.at[d.unit_id, "fault"] in DATA_FAULTS | {"none"} for d in crew)

    onset_err = []
    for d in hits:
        start = truth.at[d.unit_id, "start_time"]
        if d.onset and pd.notna(start) and truth.at[d.unit_id, "fault"] not in DATA_FAULTS:
            onset_err.append(abs((pd.Timestamp(d.onset) - pd.Timestamp(start).normalize()).days))
    out["onset_mae_days"] = float(np.mean(onset_err)) if onset_err else float("nan")
    out["tp"], out["fp"], out["fn"] = tp, fp, fn
    return out


def score_naive(flags: pd.Series, truth: pd.DataFrame) -> dict[str, float]:
    """Naive thresholding pages the crew for every flag."""
    t = truth.set_index("unit_id")["fault"]
    pos = set(t.index[t != "none"])
    flagged = set(flags.index[flags])
    tp, fp, fn = len(pos & flagged), len(flagged - pos), len(pos - flagged)
    out = {f"detect_{k}": v for k, v in _prf(tp, fp, fn).items()}
    out["false_crew_pages"] = sum(t[u] in DATA_FAULTS | {"none"} for u in flagged)
    out["tp"], out["fp"], out["fn"] = tp, fp, fn
    return out
