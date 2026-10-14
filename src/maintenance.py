"""Rule-based maintenance guidance, kept SEPARATE from the ML prediction.

Reference rules come from the AI4I 2020 dataset's own documentation of how its
failure modes were generated (please verify them on the UCI page; the TWF interval is a reference range, not a failure guarantee). The dataset is
synthetic, so real machines need real manufacturer limits. Percentile-based
checks only say "unusual compared with this dataset".
"""
from __future__ import annotations

import math

from src.data_preprocessing import DISPLAY_NAMES, UNITS

# Tool wear failure (TWF) reference interval from the dataset documentation: a TWF is described as
# occurring at a randomly chosen tool-wear time between 200 and 240 minutes (inclusive).
TWF_INTERVAL_MIN, TWF_INTERVAL_MAX = 200, 240   # minutes
HDF_TEMP_GAP_K, HDF_RPM = 8.6, 1380        # heat dissipation: small temp gap AND low speed
POWER_LOW_W, POWER_HIGH_W = 3500, 9000     # power failure outside this band
OSF_LIMITS = {"L": 11000, "M": 12000, "H": 13000}   # tool wear [min] x torque [Nm] limit by type


def power_watts(torque_nm: float, rpm: float) -> float:
    """Mechanical power P = torque x angular speed (rad/s)."""
    return torque_nm * rpm * 2 * math.pi / 60


def maintenance_checks(inputs: dict, bundle: dict, prediction: dict | None = None) -> list[dict]:
    """Return a list of {level, title, detail}; level in {alert, watch, ok, info}."""
    items = []
    tw, tq, rpm = float(inputs["tool_wear"]), float(inputs["torque"]), float(inputs["rpm"])
    gap = float(inputs["process_temp"]) - float(inputs["air_temp"])
    ranges = bundle["feature_ranges"]

    if TWF_INTERVAL_MIN <= tw <= TWF_INTERVAL_MAX:
        items.append({"level": "alert", "title": "Tool wear is inside the dataset's TWF reference interval",
                      "detail": f"Tool wear is {tw:g} min, inside the {TWF_INTERVAL_MIN}-{TWF_INTERVAL_MAX} min "
                                "interval where the dataset documentation says a tool wear failure may occur. "
                                "This does not mean a failure will happen; it is a prompt to review tool "
                                "condition and your own tool-replacement policy."})
    elif tw > TWF_INTERVAL_MAX:
        items.append({"level": "alert", "title": "Tool wear is above the dataset's TWF reference interval",
                      "detail": f"Tool wear is {tw:g} min, above the {TWF_INTERVAL_MIN}-{TWF_INTERVAL_MAX} min "
                                "interval. The dataset documentation does not describe behaviour beyond it, so "
                                "this is only a flag that wear is very high; review tool condition."})
    elif tw >= ranges["tool_wear"]["p90"]:
        items.append({"level": "watch", "title": "Tool wear is high for this dataset",
                      "detail": f"{tw:g} min is above the 90th percentile of the training data "
                                f"({ranges['tool_wear']['p90']:g} min), but below the {TWF_INTERVAL_MIN} min start "
                                "of the TWF reference interval. Consider planning an inspection."})

    limit = OSF_LIMITS[inputs["Type"]]
    load = tw * tq
    if load >= limit:
        items.append({"level": "alert", "title": "Overstrain indicator exceeded",
                      "detail": f"Tool wear x torque = {load:,.0f} min*Nm, at/above the {limit:,} reference "
                                f"for type {inputs['Type']}. Check load and tool condition."})
    elif load >= 0.9 * limit:
        items.append({"level": "watch", "title": "Approaching overstrain reference",
                      "detail": f"Tool wear x torque = {load:,.0f} min*Nm is within 10% of the {limit:,} reference."})

    if gap < HDF_TEMP_GAP_K and rpm < HDF_RPM:
        items.append({"level": "alert", "title": "Possible heat-dissipation problem",
                      "detail": f"Temperature gap is {gap:.1f} K (< {HDF_TEMP_GAP_K}) while speed is {rpm:g} rpm "
                                f"(< {HDF_RPM}). Inspect cooling and heat removal."})
    elif gap < HDF_TEMP_GAP_K:
        items.append({"level": "watch", "title": "Small process/air temperature gap",
                      "detail": f"Gap is {gap:.1f} K. Check temperature sensors and cooling."})

    p = power_watts(tq, rpm)
    if p < POWER_LOW_W or p > POWER_HIGH_W:
        items.append({"level": "alert", "title": "Mechanical power outside reference band",
                      "detail": f"Torque x speed gives about {p:,.0f} W, outside {POWER_LOW_W:,}-{POWER_HIGH_W:,} W. "
                                "Investigate drive load and speed settings."})
    elif p < POWER_LOW_W * 1.1 or p > POWER_HIGH_W * 0.9:
        items.append({"level": "watch", "title": "Power near edge of reference band",
                      "detail": f"About {p:,.0f} W is within 10% of the {POWER_LOW_W:,}-{POWER_HIGH_W:,} W band."})

    if tq > ranges["torque"]["p95"]:
        items.append({"level": "watch", "title": "Unusually high torque (vs dataset)",
                      "detail": f"{tq:g} Nm is above the dataset's 95th percentile ({ranges['torque']['p95']:g} Nm)."})
    if rpm < ranges["rpm"]["p05"]:
        items.append({"level": "watch", "title": "Unusually low rotational speed (vs dataset)",
                      "detail": f"{rpm:g} rpm is below the dataset's 5th percentile ({ranges['rpm']['p05']:g} rpm)."})

    if prediction and prediction["predicted_class"] == 1:
        items.append({"level": "alert", "title": "The ML model flagged this machine",
                      "detail": "Schedule an appropriate inspection. The model score is a screening signal, "
                                "not a diagnosis."})
    if not any(i["level"] in ("alert", "watch") for i in items):
        items.append({"level": "ok", "title": "No rule-based concerns found",
                      "detail": "None of the documented conditions were triggered for these inputs."})
    items.append({"level": "info", "title": "How to use this page",
                  "detail": "These rules are general guidance, independent of the ML model. Always follow the "
                            "machine manufacturer's limits and have a qualified engineer validate decisions."})
    return items
