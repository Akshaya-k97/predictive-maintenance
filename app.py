"""Streamlit dashboard.  Run with:  streamlit run app.py

The dashboard only LOADS the saved model (models/best_model.joblib) and the
evaluation results (reports/results.json) created by train_model.py.
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import visualizations as viz
from src.counterfactual import find_counterfactuals
from src.data_preprocessing import (DISPLAY_NAMES, FAILURE_TYPE_COLUMNS, FAILURE_TYPE_NAMES,
                                    NUMERIC_FEATURES, TARGET_COLUMN, TYPE_CATEGORIES, UNITS,
                                    load_raw_data)
from src.maintenance import maintenance_checks
from src.predict import (DEFAULT_RISK_THRESHOLDS, MODEL_PATH, ModelNotFoundError, load_bundle,
                         predict_machine, risk_category, validate_inputs)
from src.train import RESULTS_PATH

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")

DISCLAIMER = ("Educational project. Predictions come from a model trained on a public, synthetic dataset and "
              "are NOT guarantees. They require professional validation and must not be used for "
              "safety-critical decisions.")

st.markdown("""
<style>
div[data-testid="stMetric"] {background:#111C33; border:1px solid #24344F; border-radius:10px; padding:12px 16px;}
.status-pill {display:inline-block; padding:6px 16px; border-radius:999px; font-weight:600; color:white;}
</style>""", unsafe_allow_html=True)


# ---------- cached loaders ----------
@st.cache_data(show_spinner=False)
def load_dataset():
    return load_raw_data()


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_bundle()


@st.cache_data(show_spinner=False)
def get_results():
    return json.loads(RESULTS_PATH.read_text())


@st.cache_resource(show_spinner="Preparing SHAP explainer ...")
def get_explainer():
    from src import explain
    return explain.build_explainer(get_bundle())


@st.cache_resource(show_spinner="Computing SHAP values on a test sample ...")
def get_shap_sample():
    from src import explain
    b = get_bundle()
    return explain.compute_shap(b, get_explainer(), b["explain_sample_X"])


def require_dataset():
    try:
        return load_dataset()
    except FileNotFoundError as exc:
        st.error("Dataset file not found.")
        st.code(str(exc))
        st.stop()


def require_model():
    """Return (bundle, results) or show training instructions and stop."""
    try:
        bundle = get_bundle()
        results = get_results()
    except (ModelNotFoundError, FileNotFoundError):
        st.error("The trained model is missing. Train it first, then refresh this page.")
        st.markdown("1. Put `ai4i2020.csv` in the `data/` folder (see `data/README.md`).\n"
                    "2. In a terminal, from the project folder, run:")
        st.code("python train_model.py", language="bash")
        st.caption(f"Expected model file: {MODEL_PATH}")
        st.stop()
    return bundle, results


def risk_thresholds():
    st.sidebar.markdown("**Risk bands (presentation rules)**")
    medium = st.sidebar.slider("Medium risk starts at score", 0.05, 0.90, DEFAULT_RISK_THRESHOLDS["medium"], 0.05)
    high = st.sidebar.slider("High risk starts at score", 0.10, 0.95, DEFAULT_RISK_THRESHOLDS["high"], 0.05)
    if high <= medium:
        st.sidebar.warning("High must be above Medium; adjusting.")
        high = min(medium + 0.05, 0.99)
    st.sidebar.caption("Bands are adjustable display rules, not calibrated probabilities or safety limits.")
    return medium, high


# ---------- shared input widgets ----------
def reset_inputs(example):
    for k, v in example.items():
        st.session_state[f"in_{k}"] = v


def input_form(bundle):
    ex, rng = bundle["example_values"], bundle["feature_ranges"]
    for k, v in ex.items():
        st.session_state.setdefault(f"in_{k}", v)
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Machine type (L / M / H quality)", TYPE_CATEGORIES, index=TYPE_CATEGORIES.index(bundle["example_values"]["Type"]), key="in_Type")
    for col, container in zip(NUMERIC_FEATURES, [c2, c3, c1, c2, c3]):
        r = rng[col]
        label = f"{DISPLAY_NAMES[col]} [{UNITS[col]}]"
        if col in ("rpm", "tool_wear"):
            container.slider(label, int(r["min"]), int(r["max"]), step=1, key=f"in_{col}")
        else:
            container.slider(label, float(r["min"]), float(r["max"]), step=0.1, format="%.1f", key=f"in_{col}")
    return {c: st.session_state[f"in_{c}"] for c in ["Type"] + NUMERIC_FEATURES}


# ---------- pages ----------
def page_overview():
    st.title("Overview")
    df = require_dataset()
    bundle, results = require_model()
    sel = results["selected"]

    st.subheader("Historical dataset statistics")
    st.caption("These describe past records in the AI4I 2020 dataset - not predictions about any machine.")
    n_fail = int(df[TARGET_COLUMN].sum())
    c = st.columns(4)
    c[0].metric("Total records", f"{len(df):,}")
    c[1].metric("Failures", f"{n_fail:,}")
    c[2].metric("Non-failures", f"{len(df) - n_fail:,}")
    c[3].metric("Observed failure rate", f"{n_fail / len(df):.2%}")

    st.subheader("Selected model (held-out test set)")
    t = sel["test_at_threshold"]
    c = st.columns(4)
    c[0].metric("Model", sel["model"])
    c[1].metric("Imbalance handling", dict(baseline="None", class_weight="Class weights", smote="SMOTE")[sel["strategy"]])
    c[2].metric("Test recall", f"{t['recall']:.3f}")
    c[3].metric("Test F1-score", f"{t['f1']:.3f}")
    st.caption(f"Metrics use the operating threshold {sel['threshold']:.3f}, tuned on out-of-fold training data. "
               f"Model chosen by: {results['meta']['selection_rule']}.")

    left, right = st.columns(2)
    from src import explain
    left.plotly_chart(viz.fig_importance(explain.native_importance(bundle)), width="stretch")
    right.plotly_chart(viz.fig_class_distribution(df), width="stretch")
    st.warning(DISCLAIMER)


def page_risk():
    st.title("Machine Risk Assessment")
    bundle, _ = require_model()
    medium, high = risk_thresholds()
    st.caption("Enter sensor readings for ONE machine. Ranges come from the training data.")
    inputs = input_form(bundle)
    b1, b2, _ = st.columns([1, 1, 4])
    analyze = b1.button("Analyze Machine", type="primary")
    b2.button("Reset to example", on_click=reset_inputs, args=(bundle["example_values"],))

    if analyze:
        errors, warns = validate_inputs(inputs, bundle)
        if errors:
            for e in errors:
                st.error(e)
        else:
            st.session_state["last_result"] = {"inputs": dict(inputs), "prediction": predict_machine(bundle, inputs),
                                               "warnings": warns}
    res = st.session_state.get("last_result")
    if not res:
        st.info("Click **Analyze Machine** to see the model's assessment.")
        return
    if res["inputs"] != inputs:
        st.info("Inputs changed since the last analysis. Click **Analyze Machine** to update.")

    pred, shown = res["prediction"], res["inputs"]
    for w in res["warnings"]:
        st.warning(w)
    name, color = risk_category(pred["probability"], medium, high)
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(viz.fig_gauge(pred["probability"], medium, high), width="stretch")
    with c2:
        st.markdown(f"**Predicted class:** {pred['label']}")
        st.markdown(f"<span class='status-pill' style='background:{color}'>{name} risk</span>", unsafe_allow_html=True)
        st.metric("Model-estimated failure score", f"{pred['probability']:.1%}")
        st.caption(f"Class rule: Failure if score >= {pred['threshold']:.3f} (tuned threshold). The score is a model "
                   "output, not a calibrated real-world failure probability"
                   + (", and class weighting/SMOTE shifts scores upward." if bundle["strategy"] != "baseline" else "."))
    st.markdown("**Input sensor summary**")
    summary = pd.DataFrame([{"Sensor": DISPLAY_NAMES[k], "Value": shown[k], "Unit": UNITS.get(k, "")} for k in shown])
    st.dataframe(summary.astype(str), hide_index=True, width="stretch")

    st.markdown("**Why did the model say this?**")
    try:
        from src import explain
        contrib, _ = explain.local_contributions(bundle, get_explainer(), shown)
        units = explain.explanation_units(bundle)
        st.plotly_chart(viz.fig_shap_local(contrib, units), width="stretch")
        for line in explain.plain_language(contrib, units):
            st.markdown(f"- {line}")
    except ImportError as exc:
        st.warning(str(exc))
    st.warning("This prediction is not a guarantee. " + DISCLAIMER)


def page_performance():
    st.title("Model Performance")
    _, results = require_model()
    meta, sel = results["meta"], results["selected"]
    st.caption(f"Test set: {meta['n_test']} rows ({meta['test_failures']} failures). Model selection used only "
               f"{meta['cv_folds']}-fold cross-validation on the training set; the test set is for reporting.")
    if not meta["smote_run"]:
        st.warning("SMOTE experiments were skipped because imbalanced-learn was not installed or failed during training.")
    for note in meta.get("skipped", []):
        st.caption(f"Skipped: {note}")

    st.subheader("Model comparison (threshold 0.5)")
    table = viz.experiments_table(results)
    base = results["baseline"]["test"]
    st.caption(f"Majority-class baseline: accuracy {base['accuracy']:.3f}, recall {base['recall']:.3f}, "
               f"F1 {base['f1']:.3f} - high accuracy while detecting nothing is why accuracy alone misleads.")
    st.dataframe(table, hide_index=True, width="stretch")
    st.plotly_chart(viz.fig_metric_comparison(results), width="stretch")

    st.subheader("Class imbalance handling")
    st.plotly_chart(viz.fig_strategy_comparison(results), width="stretch")
    st.caption("Compared on identical cross-validation folds of the training data. Whether class weights or "
               "SMOTE help is an empirical result, not an assumption.")

    st.subheader("Confusion matrix")
    options = ["Selected model (operating threshold)"] + [e["label"] for e in results["experiments"]]
    choice = st.selectbox("Choose a model", options)
    if choice == options[0]:
        fig = viz.fig_confusion_matrix(sel["test_at_threshold"],
                                       f"{sel['label']} @ threshold {sel['threshold']:.3f}")
    else:
        exp = next(e for e in results["experiments"] if e["label"] == choice)
        fig = viz.fig_confusion_matrix(exp["test"], f"{choice} @ threshold 0.5")
    st.plotly_chart(fig, width="stretch")
    st.info("A **false negative** is a real failure the model missed (costly: unplanned downtime). A **false "
            "positive** is a false alarm (an unnecessary inspection). Because misses are usually costlier, the "
            "final model was chosen by F2 (recall weighted above precision).")

    a, b = st.columns(2)
    a.plotly_chart(viz.fig_roc(results, sel["label"]), width="stretch")
    b.plotly_chart(viz.fig_pr(results, sel["label"]), width="stretch")
    st.caption("Thick line = selected model. With rare failures, the precision-recall curve is the more "
               "informative of the two.")


def page_explainability():
    st.title("Explainability")
    bundle, _ = require_model()
    from src import explain
    tabs = st.tabs(["Global importance", "SHAP summary", "Local SHAP", "Counterfactual"])

    with tabs[0]:
        st.plotly_chart(viz.fig_importance(explain.native_importance(bundle)), width="stretch")
        st.caption("Which inputs the model relies on most overall, from the model's own internals. It shows "
                   "usage by the model, not direction of effect and not physical causation.")

    with tabs[1]:
        try:
            values, X_t, _ = get_shap_sample()
            fig = viz.shap_summary_figure(values, X_t)
            st.pyplot(fig, width="stretch")
            plt.close(fig)                       # free memory; Streamlit has already rendered it
            st.caption("Each dot is a machine from a test sample. Horizontal position = push on the model's output "
                       f"({explain.explanation_units(bundle)} units; right = towards failure). Colour = feature value "
                       "(red high, blue low).")
        except ImportError as exc:
            st.warning(str(exc))

    X_s, y_s = bundle["explain_sample_X"], bundle["explain_sample_y"]
    with tabs[2]:
        options = ["Test-set example"] + (["Last analyzed machine"] if "last_result" in st.session_state else [])
        source = st.radio("Machine to explain", options, horizontal=True)
        if source == "Last analyzed machine":
            inputs = st.session_state["last_result"]["inputs"]
        else:
            idx = st.number_input("Test sample row", 0, len(X_s) - 1, 0, 1)
            inputs = X_s.iloc[int(idx)].to_dict()
            st.caption(f"Historical label for this row: {'Failure' if y_s.iloc[int(idx)] == 1 else 'No failure'}")
        pred = predict_machine(bundle, inputs)
        st.write(f"Model output: **{pred['label']}** (score {pred['probability']:.1%})")
        try:
            contrib, base_value = explain.local_contributions(bundle, get_explainer(), inputs)
            units = explain.explanation_units(bundle)
            st.plotly_chart(viz.fig_shap_local(contrib, units), width="stretch")
            st.caption(f"Base value (average model output): {base_value:.3f} ({units} units). Red bars push towards "
                       "failure, blue bars away.")
            for line in explain.plain_language(contrib, units):
                st.markdown(f"- {line}")
        except ImportError as exc:
            st.warning(str(exc))

    with tabs[3]:
        st.markdown("**Counterfactual (what-if) explanation - hypothetical only.** The search tries small changes "
                    "to the inputs and asks the real model whether its prediction would flip.")
        source = "Last analyzed machine" if "last_result" in st.session_state else None
        if source:
            base_inputs = st.session_state["last_result"]["inputs"]
            st.caption("Using the last machine analysed on the Risk Assessment page.")
        else:
            base_inputs = bundle["example_values"]
            st.caption("No machine analysed yet; using the representative example values.")
        mutable = st.multiselect("Features allowed to change", NUMERIC_FEATURES,
                                 default=["tool_wear", "torque", "rpm"], format_func=DISPLAY_NAMES.get)
        c1, c2 = st.columns(2)
        pct = c1.slider("Max change per feature (% of observed range)", 5, 50, 30, 5)
        k = c2.slider("Max features changed together", 1, 3, 2)
        if st.button("Search for counterfactuals"):
            cf = find_counterfactuals(bundle, base_inputs, mutable, max_change_pct=pct, max_features=k)
            st.write(f"Original: **{'Failure' if cf['original_class'] else 'No Failure'}** "
                     f"(score {cf['original_probability']:.1%}); {cf['n_evaluated']} plausible scenarios tested.")
            if not cf["found"]:
                st.warning(cf["message"])
                if cf["closest_probability"] is not None:
                    st.caption(f"Closest score reached: {cf['closest_probability']:.1%}")
            else:
                rows = []
                for i, c in enumerate(cf["counterfactuals"], 1):
                    rows.append({"Scenario": i,
                                 "Proposed changes": "; ".join(
                                     f"{ch['display_name']}: {ch['original']:g} -> {ch['new']:g} {ch['unit']}"
                                     for ch in c["changes"]),
                                 "New score": f"{c['new_probability']:.1%}",
                                 "New prediction": "Failure" if c["new_class"] else "No Failure"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.info(cf["message"])
            st.caption("Type is never changed; values stay within the training range and a realistic process/air "
                       "temperature gap. Changes such as tool wear may not be physically adjustable in practice "
                       "(e.g. they need a tool replacement).")


def page_failure_types():
    st.title("Failure Type Analysis (historical)")
    df = require_dataset()
    st.caption("Descriptive analysis of the dataset's own TWF/HDF/PWF/OSF/RNF flags. The binary model does NOT "
               "predict failure types, and no failure-type classifier is implemented in this project.")
    c = st.columns(5)
    for col, code in zip(c, FAILURE_TYPE_COLUMNS):
        col.metric(FAILURE_TYPE_NAMES[code], int(df[code].sum()))
    a, b = st.columns(2)
    a.plotly_chart(viz.fig_failure_type_counts(df), width="stretch")
    b.plotly_chart(viz.fig_failure_overlap(df), width="stretch")

    flags = df[FAILURE_TYPE_COLUMNS].sum(axis=1)
    st.markdown("**Label consistency checks (computed from the data)**")
    st.write(f"- Machine failure = 1 with no type flag: **{int(((df[TARGET_COLUMN] == 1) & (flags == 0)).sum())}**")
    st.write(f"- A type flag = 1 but Machine failure = 0: **{int(((df[TARGET_COLUMN] == 0) & (flags > 0)).sum())}**")
    st.write(f"- Records with more than one type flag: **{int((flags > 1).sum())}**")
    st.caption("Flags can overlap, so they are not mutually exclusive classes, and some rows may not follow the "
               "simple 'any flag -> failure' rule. Rare types have few examples, so patterns are uncertain.")
    feature = st.selectbox("Sensor to compare across failure types", NUMERIC_FEATURES, format_func=DISPLAY_NAMES.get)
    st.plotly_chart(viz.fig_failure_type_box(df, feature), width="stretch")


def page_maintenance():
    st.title("Maintenance Insights")
    bundle, _ = require_model()
    st.caption("General rule-based guidance, separate from the ML prediction. No remaining-useful-life estimate is "
               "given: this dataset has no time-series history per machine.")
    res = st.session_state.get("last_result")
    if not res:
        st.info("Analyse a machine on the **Machine Risk Assessment** page first; its readings are checked here.")
        return
    icons = {"alert": "🔴", "watch": "🟠", "ok": "🟢", "info": "ℹ️"}
    for item in maintenance_checks(res["inputs"], bundle, res["prediction"]):
        with st.container(border=True):
            st.markdown(f"{icons[item['level']]} **{item['title']}**")
            st.write(item["detail"])
    st.caption("Reference values come from the AI4I 2020 dataset documentation (synthetic data) and dataset "
               "percentiles. They are not manufacturer limits. Do not read SHAP results as proof that a sensor caused "
               "a failure.")


PAGES = {"Overview": page_overview, "Machine Risk Assessment": page_risk, "Model Performance": page_performance,
         "Explainability": page_explainability, "Failure Type Analysis": page_failure_types,
         "Maintenance Insights": page_maintenance}


def main():
    st.sidebar.title("🛠️ Predictive Maintenance")
    page = st.sidebar.radio("Navigate", list(PAGES))
    st.sidebar.caption(DISCLAIMER)
    PAGES[page]()


main()
