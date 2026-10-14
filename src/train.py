"""Model training, fair comparison and model saving.

Design (important for the viva):
1. Split first (stratified). The test set is locked away.
2. For every (model x imbalance strategy) pair, run 5-fold stratified
   cross-validation on the TRAINING data only. Preprocessing and SMOTE live
   inside the pipeline, so they are re-fitted on each training fold only.
3. The final model is chosen by cross-validated F2 (recall-weighted), not accuracy.
4. The decision threshold is tuned on out-of-fold TRAINING predictions.
5. Only then is the test set used - for reporting, once per model.
"""
from __future__ import annotations

import argparse
import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import (DATA_PATH, FEATURE_COLUMNS, NUMERIC_FEATURES, PROJECT_ROOT,
                                    RANDOM_SEED, TYPE_CATEGORIES, build_preprocessor,
                                    data_quality_report, load_raw_data, split_data)
from src.evaluate import best_threshold, compute_metrics, curve_data

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
RESULTS_PATH = PROJECT_ROOT / "reports" / "results.json"
COMPARISON_CSV = PROJECT_ROOT / "reports" / "model_comparison.csv"

CV_FOLDS = 5
STRATEGIES = {"baseline": "No imbalance handling", "class_weight": "Class weights", "smote": "SMOTE"}
# GradientBoostingClassifier has no class_weight option, so that combo is skipped.
NO_CLASS_WEIGHT = {"Gradient Boosting"}

SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
    "roc_auc": "roc_auc",
}


def print_versions() -> None:
    """Print library versions so any error report is easy to reproduce."""
    import importlib
    parts = [f"python {platform.python_version()}"]
    for name in ("numpy", "pandas", "sklearn", "imblearn", "shap", "streamlit", "plotly"):
        try:
            parts.append(f"{name} {importlib.import_module(name).__version__}")
        except Exception:
            parts.append(f"{name} (not installed)")
    print("Environment: " + ", ".join(parts))


def get_models() -> dict:
    """Fresh, unfitted classifiers (modest, untuned settings - easy to explain)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                n_jobs=-1, random_state=RANDOM_SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }


def smote_available() -> bool:
    try:
        import imblearn  # noqa: F401
        return True
    except ImportError:
        return False


def build_pipeline(model_name: str, strategy: str):
    """preprocessing -> [SMOTE] -> classifier, all fitted on training data only."""
    clf = get_models()[model_name]
    if strategy == "class_weight":
        clf.set_params(class_weight="balanced")
    steps = [("prep", build_preprocessor())]
    if strategy == "smote":
        from imblearn.over_sampling import SMOTENC
        from imblearn.pipeline import Pipeline as ImbPipeline
        # Column 0 is the encoded machine Type -> treat it as categorical.
        steps.append(("smote", SMOTENC(categorical_features=[0], random_state=RANDOM_SEED)))
        return ImbPipeline(steps + [("clf", clf)])
    return Pipeline(steps + [("clf", clf)])


def _round_floats(d: dict, ndigits: int = 4) -> dict:
    return {k: (round(v, ndigits) if isinstance(v, float) else v) for k, v in d.items()}


def run_training(data_path: Path | str = DATA_PATH) -> dict:
    warnings.filterwarnings("ignore", category=UserWarning)
    print("1/6 Loading and checking data ...")
    df = load_raw_data(data_path)
    quality = data_quality_report(df)
    print(f"    rows={quality['n_rows']}  features={quality['n_features']}  "
          f"missing={quality['missing_total']}  duplicates={quality['n_duplicates']}  "
          f"failure_rate={quality['failure_rate']:.3%}")

    print("2/6 Stratified train/test split (test set is now locked) ...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"    train={len(X_train)}  test={len(X_test)}  "
          f"train failures={int(y_train.sum())}  test failures={int(y_test.sum())}")

    print_versions()
    use_smote = smote_available()
    if not use_smote:
        print("    WARNING: imbalanced-learn not installed -> SMOTE experiments skipped. "
              "Run: pip install imbalanced-learn")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    print("3/6 Majority-class baseline ...")
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    dummy_proba = dummy.predict_proba(X_test)[:, 1]
    baseline = {"name": "Majority-class baseline",
                "test": _round_floats(compute_metrics(y_test, dummy_proba, 0.5))}

    print(f"4/6 Cross-validating models ({CV_FOLDS}-fold, training data only) ...")
    experiments, fitted, skipped = [], {}, []
    for model_name in get_models():
        for strategy, strategy_label in STRATEGIES.items():
            if strategy == "class_weight" and model_name in NO_CLASS_WEIGHT:
                continue
            if strategy == "smote" and not use_smote:
                continue
            try:
                pipe = build_pipeline(model_name, strategy)
                cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING,
                                        n_jobs=1, error_score="raise")
                pipe.fit(X_train, y_train)                   # fit on full training set
            except Exception as exc:
                if strategy != "smote":
                    raise                                    # real bug: do not hide it
                # A scikit-learn / imbalanced-learn version mismatch should not stop the whole run.
                note = f"{model_name} | SMOTE skipped ({type(exc).__name__}: {exc})"
                print(f"    WARNING: {note}")
                skipped.append(note)
                continue
            proba = pipe.predict_proba(X_test)[:, 1]         # test set: REPORTING only
            curves = curve_data(y_test, proba)
            exp = {
                "model": model_name, "strategy": strategy, "strategy_label": strategy_label,
                "label": f"{model_name} | {strategy_label}",
                "cv": {f"{m}_mean": round(float(cv_res[f"test_{m}"].mean()), 4) for m in SCORING}
                      | {f"{m}_std": round(float(cv_res[f"test_{m}"].std()), 4) for m in SCORING},
                "test": _round_floats(compute_metrics(y_test, proba, 0.5)),
                **curves,
            }
            experiments.append(exp)
            fitted[(model_name, strategy)] = pipe
            print(f"    {exp['label']:<48} CV F2={exp['cv']['f2_mean']:.3f}  "
                  f"test recall={exp['test']['recall']:.3f}  test F1={exp['test']['f1']:.3f}")

    print("5/6 Selecting final model by cross-validated F2 (training data only) ...")
    best = max(experiments, key=lambda e: e["cv"]["f2_mean"])
    pipe = fitted[(best["model"], best["strategy"])]
    oof = cross_val_predict(clone(pipe), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold, oof_f2 = best_threshold(y_train, oof, beta=2.0)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    selected_test = _round_floats(compute_metrics(y_test, test_proba, threshold))
    print(f"    selected: {best['label']}")
    print(f"    operating threshold (tuned on out-of-fold training data): {threshold:.3f}")
    print(f"    test @ threshold: recall={selected_test['recall']:.3f} "
          f"precision={selected_test['precision']:.3f} F1={selected_test['f1']:.3f}")

    print("6/6 Saving model bundle and reports ...")
    ranges = {}
    for col in NUMERIC_FEATURES:
        s = X_train[col]
        ranges[col] = {"min": float(s.min()), "max": float(s.max()), "median": float(s.median()),
                       "p05": float(s.quantile(0.05)), "p90": float(s.quantile(0.90)),
                       "p95": float(s.quantile(0.95))}
    diff = X_train["process_temp"] - X_train["air_temp"]
    example = {"Type": X_train["Type"].mode()[0]}
    for col in NUMERIC_FEATURES:
        example[col] = int(round(ranges[col]["median"])) if col in ("rpm", "tool_wear") \
            else round(ranges[col]["median"], 1)

    sample_idx = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_SEED).index
    prep = pipe.named_steps["prep"]
    background = prep.transform(X_train.sample(n=min(200, len(X_train)), random_state=RANDOM_SEED))

    bundle = {
        "pipeline": pipe,
        "model_name": best["model"], "strategy": best["strategy"],
        "strategy_label": best["strategy_label"],
        "threshold": threshold,
        "feature_columns": FEATURE_COLUMNS, "type_categories": TYPE_CATEGORIES,
        "feature_ranges": ranges,
        "temp_diff_range": {"min": float(diff.min()), "max": float(diff.max()),
                            "p05": float(diff.quantile(0.05))},
        "example_values": example,
        "explain_sample_X": X_test.loc[sample_idx].reset_index(drop=True),
        "explain_sample_y": y_test.loc[sample_idx].reset_index(drop=True),
        "background": background,
        "sklearn_version": sklearn.__version__, "seed": RANDOM_SEED,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    results = {
        "meta": {
            "quality": quality, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "train_failures": int(y_train.sum()), "test_failures": int(y_test.sum()),
            "seed": RANDOM_SEED, "cv_folds": CV_FOLDS, "smote_run": use_smote and any(e["strategy"] == "smote" for e in experiments),
            "skipped": skipped,
            "selection_rule": "highest mean 5-fold CV F2 score on the training set",
            "threshold_rule": "maximise F2 on out-of-fold training predictions",
            "python": platform.python_version(), "sklearn": sklearn.__version__,
        },
        "baseline": baseline,
        "experiments": experiments,
        "selected": {"label": best["label"], "model": best["model"], "strategy": best["strategy"],
                     "threshold": threshold, "oof_f2": round(oof_f2, 4),
                     "test_at_threshold": selected_test},
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    rows = [{"model": e["model"], "imbalance_handling": e["strategy_label"],
             **{f"cv_{k}": v for k, v in e["cv"].items() if k.endswith("_mean")},
             **{f"test_{k}": v for k, v in e["test"].items()
                if k in ("precision", "recall", "f1", "roc_auc", "accuracy")}} for e in experiments]
    pd.DataFrame(rows).to_csv(COMPARISON_CSV, index=False)
    print(f"    saved {MODEL_PATH.relative_to(PROJECT_ROOT)}, "
          f"{RESULTS_PATH.relative_to(PROJECT_ROOT)}, {COMPARISON_CSV.relative_to(PROJECT_ROOT)}")
    print("Done. Launch the dashboard with:  streamlit run app.py")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the predictive-maintenance models.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to ai4i2020.csv")
    args = parser.parse_args()
    run_training(args.data)


if __name__ == "__main__":
    main()
