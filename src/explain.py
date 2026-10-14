"""Explainability: model-native importance, SHAP values, plain-language text.

SHAP shows how the MODEL's output depends on each input. It describes the
model's behaviour, not physical cause and effect in the machine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import DISPLAY_NAMES, FEATURE_COLUMNS, UNITS
from src.predict import to_frame


def _import_shap():
    try:
        import shap
        return shap
    except ImportError as exc:
        raise ImportError("SHAP is not installed. Run: pip install shap") from exc


def native_importance(bundle: dict) -> pd.DataFrame:
    """Importance straight from the model (impurity importance or |coefficient|)."""
    clf = bundle["pipeline"].named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values, kind = np.asarray(clf.feature_importances_), "Impurity-based importance"
    elif hasattr(clf, "coef_"):
        values, kind = np.abs(np.ravel(clf.coef_)), "|Coefficient| (features are standardised)"
    else:
        raise ValueError("This model type has no built-in feature importance.")
    df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": values})
    df["display_name"] = df["feature"].map(DISPLAY_NAMES)
    df.attrs["kind"] = kind
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def build_explainer(bundle: dict):
    """Create a SHAP explainer that matches the selected model type."""
    shap = _import_shap()
    clf = bundle["pipeline"].named_steps["clf"]
    if isinstance(clf, (RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier)):
        return shap.TreeExplainer(clf)
    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, bundle["background"])
    return shap.Explainer(clf.predict_proba, bundle["background"])


def explanation_units(bundle: dict) -> str:
    """Random Forest / Decision Tree SHAP is in probability units; boosting and
    logistic regression are in log-odds units."""
    clf = bundle["pipeline"].named_steps["clf"]
    if isinstance(clf, (RandomForestClassifier, DecisionTreeClassifier)):
        return "probability"
    return "log-odds"


def _positive_class(raw):
    """Different SHAP versions return different shapes; pick failure-class values."""
    if isinstance(raw, list):
        return np.asarray(raw[1])
    raw = np.asarray(raw)
    if raw.ndim == 3:
        return raw[:, :, 1]
    return raw


def compute_shap(bundle: dict, explainer, X_raw: pd.DataFrame):
    """Return (shap_values[n, features], transformed_X DataFrame, base_value)."""
    X_t = pd.DataFrame(bundle["pipeline"].named_steps["prep"].transform(X_raw[FEATURE_COLUMNS]),
                       columns=FEATURE_COLUMNS)
    values = _positive_class(explainer.shap_values(X_t.values))
    if values.shape != X_t.shape:       # catches SHAP-version differences in output layout early
        raise ValueError(f"Unexpected SHAP output shape {values.shape}; expected {X_t.shape}. "
                         "Please report your shap and scikit-learn versions.")
    base = np.ravel(explainer.expected_value)
    base_value = float(base[1] if len(base) > 1 else base[0])
    return values, X_t, base_value


def local_contributions(bundle: dict, explainer, inputs: dict):
    """Per-feature SHAP contributions for one machine -> (DataFrame, base_value)."""
    X_raw = to_frame(inputs)
    values, _, base_value = compute_shap(bundle, explainer, X_raw)
    df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "display_name": [DISPLAY_NAMES[c] for c in FEATURE_COLUMNS],
        "value": [inputs[c] for c in FEATURE_COLUMNS],
        "shap_value": values[0],
    })
    df["abs"] = df["shap_value"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True), base_value


def describe_value(feature: str, value) -> str:
    return f"{value}" if feature == "Type" else f"{value:g} {UNITS[feature]}"


def plain_language(contrib: pd.DataFrame, units: str, top_n: int = 3) -> list[str]:
    """Beginner-friendly sentences about the biggest contributions."""
    lines = []
    for _, row in contrib.head(top_n).iterrows():
        direction = "pushed the model's score UP (towards failure)" if row["shap_value"] > 0 \
            else "pushed the model's score DOWN (towards no failure)"
        lines.append(f"**{row['display_name']}** = {describe_value(row['feature'], row['value'])} "
                     f"{direction} by {abs(row['shap_value']):.3f} ({units} units).")
    lines.append("These describe how the *model* uses each input for this machine. They show association "
                 "learned from data, not proof that a sensor *caused* a failure.")
    return lines
