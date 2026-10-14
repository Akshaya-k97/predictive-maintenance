"""Counterfactual ("what-if") search using the real trained model.

Idea: starting from a machine's inputs, try small changes to a few inputs and
keep the smallest changes that make the model's predicted class flip.
These are HYPOTHETICAL scenarios about the model, not guaranteed actions.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.data_preprocessing import DISPLAY_NAMES, FEATURE_COLUMNS, NUMERIC_FEATURES, UNITS
from src.predict import to_frame

DECIMALS = {"air_temp": 1, "process_temp": 1, "rpm": 0, "torque": 1, "tool_wear": 0}


def _candidate_values(feature, original, bundle, max_change_pct, steps_per_side):
    r = bundle["feature_ranges"][feature]
    span = r["max"] - r["min"]
    cap = span * max_change_pct / 100.0
    offsets = np.linspace(-cap, cap, 2 * steps_per_side + 1)
    vals = np.round(np.clip(original + offsets, r["min"], r["max"]), DECIMALS[feature])
    vals = np.unique(vals)
    return vals[~np.isclose(vals, original)]          # must actually change


def find_counterfactuals(bundle: dict, inputs: dict, mutable_features=None,
                         max_change_pct: float = 30.0, steps_per_side: int = 7,
                         max_features: int = 2, top_k: int = 3) -> dict:
    """Search for the smallest plausible changes that flip the predicted class.

    Plausibility rules: Type is never changed; every value stays inside the
    range seen in training; process-air temperature gap stays inside the
    observed range; each change is capped at `max_change_pct` % of the
    feature's observed range.
    """
    pipe, thr = bundle["pipeline"], float(bundle["threshold"])
    mutable = [f for f in (mutable_features or NUMERIC_FEATURES) if f in NUMERIC_FEATURES]
    original = {c: inputs[c] for c in FEATURE_COLUMNS}
    p0 = float(pipe.predict_proba(to_frame(original))[0, 1])
    flagged0 = p0 >= thr
    out = {"original_probability": p0, "original_class": int(flagged0), "threshold": thr,
           "target_class": int(not flagged0), "found": False, "counterfactuals": [],
           "n_evaluated": 0, "closest_probability": None, "message": ""}
    if not mutable:
        out["message"] = "No features were selected for the search."
        return out

    grids = {f: _candidate_values(f, float(original[f]), bundle, max_change_pct, steps_per_side)
             for f in mutable}
    frames, combos = [], []
    for k in range(1, max_features + 1):
        for combo in combinations(mutable, k):
            mesh = np.array(np.meshgrid(*[grids[f] for f in combo], indexing="ij")).reshape(len(combo), -1).T
            if len(mesh) == 0:
                continue
            data = {c: np.full(len(mesh), original[c]) for c in FEATURE_COLUMNS}
            for j, f in enumerate(combo):
                data[f] = mesh[:, j]
            frames.append(pd.DataFrame(data))
            combos += [combo] * len(mesh)
    if not frames:
        out["message"] = "No candidate changes could be generated."
        return out

    cand = pd.concat(frames, ignore_index=True)
    combos = np.array([",".join(c) for c in combos])

    d = bundle["temp_diff_range"]
    gap = cand["process_temp"] - cand["air_temp"]
    ok = ((gap >= d["min"]) & (gap <= d["max"])).values        # realistic temperature gap
    cand, combos = cand[ok].reset_index(drop=True), combos[ok]
    if cand.empty:
        out["message"] = "No plausible candidates remained after applying the plausibility rules."
        return out

    proba = pipe.predict_proba(cand[FEATURE_COLUMNS])[:, 1]    # real model, every candidate
    out["n_evaluated"] = int(len(cand))
    out["closest_probability"] = float(proba.min() if flagged0 else proba.max())

    spans = {f: bundle["feature_ranges"][f]["max"] - bundle["feature_ranges"][f]["min"] for f in NUMERIC_FEATURES}
    cost = sum((cand[f] - float(original[f])).abs() / spans[f] for f in mutable)
    flips = (proba >= thr) != flagged0
    if not flips.any():
        out["message"] = ("No counterfactual found within the search limits "
                          f"({int(len(cand))} plausible scenarios tested). This does not mean none exists "
                          "- try allowing larger changes or more features.")
        return out

    table = pd.DataFrame({"combo": combos, "cost": cost, "proba": proba, "idx": np.arange(len(cand))})
    table = table[flips].sort_values("cost").drop_duplicates("combo").head(top_k)
    for _, row in table.iterrows():
        values = cand.iloc[int(row["idx"])]
        changes = []
        for f in row["combo"].split(","):
            changes.append({"feature": f, "display_name": DISPLAY_NAMES[f], "unit": UNITS[f],
                            "original": float(original[f]), "new": float(values[f]),
                            "delta": float(values[f] - float(original[f]))})
        out["counterfactuals"].append({"changes": changes, "new_probability": float(row["proba"]),
                                       "new_class": int(row["proba"] >= thr), "size": float(row["cost"])})
    out["found"] = True
    out["message"] = ("Hypothetical scenarios only: they show what would change the MODEL's prediction, "
                      "not guaranteed or validated maintenance actions.")
    return out
