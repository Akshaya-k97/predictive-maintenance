"""Loading the saved model and making validated predictions."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import DISPLAY_NAMES, FEATURE_COLUMNS, NUMERIC_FEATURES, PROJECT_ROOT, UNITS

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

# Risk bands are PRESENTATION RULES only (configurable in the dashboard sidebar).
# They are not calibrated failure probabilities and not industrial safety limits.
DEFAULT_RISK_THRESHOLDS = {"medium": 0.20, "high": 0.50}
RISK_COLORS = {"Low": "#22C55E", "Medium": "#F59E0B", "High": "#EF4444"}


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file does not exist yet."""


def load_bundle(path: Path | str = MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            f"No trained model found at '{path}'.\n"
            "Train the models first (from the project folder):\n"
            "    python train_model.py"
        )
    return joblib.load(path)


def validate_inputs(inputs: dict, bundle: dict):
    """Return (errors, warnings). Values must lie inside the range seen in training."""
    errors, warnings = [], []
    if inputs.get("Type") not in bundle["type_categories"]:
        errors.append(f"Machine type must be one of {bundle['type_categories']}.")
    for col in NUMERIC_FEATURES:
        try:
            value = float(inputs[col])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{DISPLAY_NAMES[col]} must be a number.")
            continue
        if not np.isfinite(value):
            errors.append(f"{DISPLAY_NAMES[col]} must be a finite number.")
            continue
        r = bundle["feature_ranges"][col]
        if not r["min"] <= value <= r["max"]:
            errors.append(f"{DISPLAY_NAMES[col]} must be between {r['min']:g} and {r['max']:g} "
                          f"{UNITS[col]} (the range the model was trained on).")
    if not errors:
        diff = float(inputs["process_temp"]) - float(inputs["air_temp"])
        d = bundle["temp_diff_range"]
        if not d["min"] <= diff <= d["max"]:
            warnings.append(f"Process - air temperature = {diff:.1f} K is outside the range seen in the "
                            f"data ({d['min']:.1f} to {d['max']:.1f} K); treat the result with extra caution.")
    return errors, warnings


def to_frame(inputs: dict) -> pd.DataFrame:
    """One-row DataFrame with the exact column order used in training."""
    return pd.DataFrame([{c: inputs[c] for c in FEATURE_COLUMNS}])


def predict_machine(bundle: dict, inputs: dict) -> dict:
    """Predict with the saved pipeline (preprocessing is inside the pipeline)."""
    proba = float(bundle["pipeline"].predict_proba(to_frame(inputs))[0, 1])
    threshold = float(bundle["threshold"])
    flagged = proba >= threshold
    return {"probability": proba, "threshold": threshold, "predicted_class": int(flagged),
            "label": "Failure" if flagged else "No Failure"}


def risk_category(probability: float, medium: float, high: float):
    """Map a model score to Low / Medium / High using configurable cut-offs."""
    if probability >= high:
        name = "High"
    elif probability >= medium:
        name = "Medium"
    else:
        name = "Low"
    return name, RISK_COLORS[name]
