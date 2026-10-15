"""Plotly chart builders (dark-navy industrial theme). Each returns a Figure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_preprocessing import (DISPLAY_NAMES, FAILURE_TYPE_COLUMNS, FAILURE_TYPE_NAMES,
                                    NUMERIC_FEATURES, TARGET_COLUMN, TYPE_CATEGORIES, UNITS)

NAVY, PANEL, GRID = "#0B1426", "#111C33", "#24344F"
BLUE, LIGHT_BLUE, GREEN, AMBER, RED, TEXT = "#3B82F6", "#60A5FA", "#22C55E", "#F59E0B", "#EF4444", "#E5E7EB"
STATUS_COLORS = {"No failure": BLUE, "Failure": RED}


def apply_theme(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=13), height=height, margin=dict(l=40, r=20, t=50 if title else 20, b=40),
        title=dict(text=title, font=dict(size=16)) if title else None,
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def _status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Status"] = np.where(out[TARGET_COLUMN] == 1, "Failure", "No failure")
    return out


def fig_class_distribution(df: pd.DataFrame) -> go.Figure:
    counts = _status(df)["Status"].value_counts().reindex(["No failure", "Failure"])
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, text=counts.values, textposition="outside",
                           marker_color=[STATUS_COLORS[i] for i in counts.index]))
    fig.update_yaxes(title="Records")
    return apply_theme(fig, "Failure vs non-failure records (historical data)")


def fig_failure_rate_by_type(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("Type")[TARGET_COLUMN].agg(["mean", "sum", "count"]).reindex(TYPE_CATEGORIES)
    fig = go.Figure(go.Bar(x=g.index, y=g["mean"] * 100, marker_color=BLUE,
                           text=[f"{m:.1f}%" for m in g["mean"] * 100], textposition="outside",
                           customdata=np.stack([g["sum"], g["count"]], axis=1),
                           hovertemplate="Type %{x}<br>%{customdata[0]} failures of %{customdata[1]}<extra></extra>"))
    fig.update_yaxes(title="Failure rate (%)")
    fig.update_xaxes(title="Machine type")
    return apply_theme(fig, "Observed failure rate by machine type")


def fig_feature_hist(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(_status(df), x=col, color="Status", barmode="overlay", opacity=0.65,
                       histnorm="probability density", nbins=40, color_discrete_map=STATUS_COLORS)
    fig.update_xaxes(title=f"{DISPLAY_NAMES[col]} [{UNITS[col]}]")
    fig.update_yaxes(title="Density")
    return apply_theme(fig, f"{DISPLAY_NAMES[col]} distribution", height=300)


def fig_correlation(df: pd.DataFrame) -> go.Figure:
    d = df[["Type"] + NUMERIC_FEATURES + [TARGET_COLUMN]].copy()
    d["Type"] = d["Type"].map({"L": 0, "M": 1, "H": 2})            # ordinal quality
    d = d.rename(columns={**DISPLAY_NAMES, TARGET_COLUMN: "Machine failure"})
    corr = d.corr().round(2)
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                               colorscale="RdBu_r", text=corr.values, texttemplate="%{text}"))
    return apply_theme(fig, "Feature correlation (Pearson)", height=460)


def fig_scatter(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    d = _status(df).sort_values("Status", ascending=False)           # draw failures on top
    fig = px.scatter(d, x=x, y=y, color="Status", opacity=0.6, color_discrete_map=STATUS_COLORS)
    fig.update_xaxes(title=f"{DISPLAY_NAMES[x]} [{UNITS[x]}]")
    fig.update_yaxes(title=f"{DISPLAY_NAMES[y]} [{UNITS[y]}]")
    return apply_theme(fig, f"{DISPLAY_NAMES[y]} vs {DISPLAY_NAMES[x]}", height=420)


def fig_box_by_status(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.box(_status(df), x="Status", y=col, color="Status", color_discrete_map=STATUS_COLORS,
                 category_orders={"Status": ["No failure", "Failure"]})
    fig.update_yaxes(title=f"{DISPLAY_NAMES[col]} [{UNITS[col]}]")
    fig.update_layout(showlegend=False)
    return apply_theme(fig, DISPLAY_NAMES[col], height=300)


def fig_importance(imp: pd.DataFrame, title: str | None = None) -> go.Figure:
    d = imp.sort_values("importance")
    fig = go.Figure(go.Bar(x=d["importance"], y=d["display_name"], orientation="h", marker_color=BLUE))
    fig.update_xaxes(title=imp.attrs.get("kind", "Importance"))
    return apply_theme(fig, title or "Global feature importance (selected model)")


def experiments_table(results: dict) -> pd.DataFrame:
    rows = []
    for e in results["experiments"]:
        rows.append({"Model": e["model"], "Imbalance handling": e["strategy_label"],
                     "CV F2": e["cv"]["f2_mean"], "CV recall": e["cv"]["recall_mean"],
                     "Precision": e["test"]["precision"], "Recall": e["test"]["recall"],
                     "F1": e["test"]["f1"], "ROC-AUC": e["test"]["roc_auc"],
                     "Accuracy": e["test"]["accuracy"], "FN": e["test"]["fn"], "FP": e["test"]["fp"]})
    return pd.DataFrame(rows)


def fig_metric_comparison(results: dict) -> go.Figure:
    t = experiments_table(results)
    t["Setup"] = t["Model"] + "<br>" + t["Imbalance handling"]
    fig = go.Figure()
    for metric, color in [("Precision", LIGHT_BLUE), ("Recall", RED), ("F1", BLUE), ("ROC-AUC", GREEN)]:
        fig.add_bar(x=t["Setup"], y=t[metric], name=metric, marker_color=color)
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.05], title="Score (test set, threshold 0.5)")
    return apply_theme(fig, "Model comparison on the held-out test set", height=460)


def fig_strategy_comparison(results: dict) -> go.Figure:
    fig = go.Figure()
    colors = {"baseline": LIGHT_BLUE, "class_weight": BLUE, "smote": AMBER}
    for strategy in ["baseline", "class_weight", "smote"]:
        rows = [e for e in results["experiments"] if e["strategy"] == strategy]
        if rows:
            fig.add_bar(x=[e["model"] for e in rows], y=[e["cv"]["f2_mean"] for e in rows],
                        error_y=dict(type="data", array=[e["cv"]["f2_std"] for e in rows]),
                        name=rows[0]["strategy_label"], marker_color=colors[strategy])
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Cross-validated F2 (training data)", range=[0, 1.05])
    return apply_theme(fig, "Imbalance handling: cross-validated F2 (mean +/- std)")

def fig_confusion_matrix(m: dict, title: str) -> go.Figure:
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    text = [[f"{labels[i][j]}<br>{z[i][j]}" for j in range(2)] for i in range(2)]
    # Scale the color axis to the *smallest* off-diagonal max, so small cells remain visible
    vmax = max(z[0][1], z[1][0], z[1][1], 1)
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted: No failure", "Predicted: Failure"],
                               y=["Actual: No failure", "Actual: Failure"], text=text,
                               texttemplate="%{text}", colorscale="Blues", showscale=False,
                               zmin=0, zmax=vmax))
    fig.update_yaxes(autorange="reversed")
    return apply_theme(fig, title, height=360)

def fig_roc(results: dict, selected_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=GRID, dash="dash"))
    for e in results["experiments"]:
        sel = e["label"] == selected_label
        fig.add_scatter(x=e["roc"]["fpr"], y=e["roc"]["tpr"], mode="lines", name=e["label"],
                        line=dict(width=4 if sel else 1.5))
    fig.update_xaxes(title="False positive rate")
    fig.update_yaxes(title="True positive rate (recall)")
    return apply_theme(fig, "ROC curves (test set)", height=460)


def fig_pr(results: dict, selected_label: str) -> go.Figure:
    fig = go.Figure()
    base = results["meta"]["test_failures"] / results["meta"]["n_test"]
    fig.add_shape(type="line", x0=0, y0=base, x1=1, y1=base, line=dict(color=GRID, dash="dash"))
    for e in results["experiments"]:
        sel = e["label"] == selected_label
        fig.add_scatter(x=e["pr"]["recall"], y=e["pr"]["precision"], mode="lines", name=e["label"],
                        line=dict(width=4 if sel else 1.5))
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    return apply_theme(fig, "Precision-recall curves (test set; dashed = no-skill)", height=460)


def fig_gauge(probability: float, medium: float, high: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100, number=dict(suffix="%", valueformat=".1f"),
        title=dict(text="Model failure score"),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=TEXT, thickness=0.25),
                   steps=[dict(range=[0, medium * 100], color="#14532D"),
                          dict(range=[medium * 100, high * 100], color="#78350F"),
                          dict(range=[high * 100, 100], color="#7F1D1D")])))
    return apply_theme(fig, height=300)


def fig_shap_local(contrib: pd.DataFrame, units: str) -> go.Figure:
    d = contrib.iloc[::-1]
    labels = [f"{n} = {v}" if f == "Type" else f"{n} = {v:g}"
              for n, f, v in zip(d["display_name"], d["feature"], d["value"])]
    fig = go.Figure(go.Bar(x=d["shap_value"], y=labels, orientation="h",
                           marker_color=[RED if v > 0 else BLUE for v in d["shap_value"]]))
    fig.update_xaxes(title=f"SHAP value ({units} units): right = towards failure, left = away")
    return apply_theme(fig, "Local explanation for this machine")


def shap_summary_figure(values, X_t: pd.DataFrame):
    """Matplotlib SHAP beeswarm on the dark theme (rendered with st.pyplot)."""
    import matplotlib.pyplot as plt
    import shap
    X_named = X_t.rename(columns=DISPLAY_NAMES)
    with plt.style.context("dark_background"):
        plt.figure(figsize=(8, 4.5))
        shap.summary_plot(values, X_named, show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor(NAVY)
        for ax in fig.axes:
            ax.set_facecolor(NAVY)
        fig.tight_layout()
    return fig


def fig_failure_type_counts(df: pd.DataFrame) -> go.Figure:
    counts = df[FAILURE_TYPE_COLUMNS].sum()
    fig = go.Figure(go.Bar(x=[FAILURE_TYPE_NAMES[c] for c in counts.index], y=counts.values,
                           text=counts.values, textposition="outside", marker_color=BLUE))
    fig.update_yaxes(title="Records with flag = 1")
    return apply_theme(fig, "Historical counts by failure type")


def fig_failure_overlap(df: pd.DataFrame) -> go.Figure:
    f = df[FAILURE_TYPE_COLUMNS].astype(int)
    co = f.T.dot(f)                                    # co-occurrence counts
    fig = go.Figure(go.Heatmap(z=co.values, x=co.columns, y=co.index, text=co.values,
                               texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed")
    return apply_theme(fig, "How often flags occur together (diagonal = total)", height=380)


def fig_failure_type_box(df: pd.DataFrame, col: str) -> go.Figure:
    parts = []
    for c in FAILURE_TYPE_COLUMNS:
        sub = df[df[c] == 1][[col]].assign(Group=c)
        parts.append(sub)
    parts.append(df[df[TARGET_COLUMN] == 0][[col]].assign(Group="No failure"))
    d = pd.concat(parts)
    fig = px.box(d, x="Group", y=col, color="Group")
    fig.update_yaxes(title=f"{DISPLAY_NAMES[col]} [{UNITS[col]}]")
    fig.update_layout(showlegend=False)
    return apply_theme(fig, f"{DISPLAY_NAMES[col]} by failure type", height=320)
