"""DataDrift - Analysis / Data Science module (Member 3)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pipeline import EXPECTED_SCHEMA

try:
    from sklearn.ensemble import IsolationForest
except ImportError:  # falls back to z-score
    IsolationForest = None

WEIGHTS = {"completeness": .30, "uniqueness": .25, "validity": .25, "schema": .20}
_by = lambda inc, t: [i for i in inc if i["type"] == t]


def quality_score(incidents, n_rows, n_cols=len(EXPECTED_SCHEMA)):
    null_cells = sum(i["evidence"]["null_count"] for i in _by(incidents, "missing_values"))
    dup = sum(i["affected_rows"] for i in _by(incidents, "duplicate_records"))
    bad = sum(i["affected_rows"] for i in _by(incidents, "value_inconsistency"))
    s_issues = 0
    for i in _by(incidents, "schema_drift"):
        e = i["evidence"]
        s_issues += len(e["missing_columns"]) if "missing_columns" in e else 1
    dims = {"completeness": 1 - null_cells / max(n_rows * n_cols, 1),
            "uniqueness": 1 - dup / max(n_rows, 1),
            "validity": 1 - bad / max(n_rows, 1),
            "schema": 1 - min(s_issues / n_cols, 1)}
    dims = {k: round(100 * max(v, 0), 1) for k, v in dims.items()}
    return {"overall": round(sum(dims[k] * w for k, w in WEIGHTS.items()), 1), "dimensions": dims}


def error_rates(incidents, n_rows):
    return {i["id"] + " " + i["title"]: round(100 * i["affected_rows"] / max(n_rows, 1), 2) for i in incidents}


def revenue_impact(raw, clean, incidents):
    col = "order_total"
    if col not in raw or col not in clean:
        return None
    reported, true = float(raw[col].sum()), float(clean[col].sum())
    dup = sum(i["evidence"].get("revenue_inflated", 0) for i in _by(incidents, "duplicate_records"))
    spike = sum(i["evidence"].get("excess_revenue", 0) for i in _by(incidents, "value_inconsistency"))
    return {"reported": reported, "corrected": true, "difference": reported - true,
            "difference_pct": round(100 * (reported - true) / true, 2) if true else 0,
            "from_duplicates": dup, "from_bad_totals": spike,
            "other": reported - true - dup - spike}


def detect_anomalies(df, sensitivity=0.05, z_thresh=3.0, iqr_factor=1.5):
    """Three detectors vote; a row is an anomaly when at least 2 of 3 agree."""
    d = df.copy()
    feats = [c for c in ["quantity", "unit_price", "order_total"] if c in d and pd.api.types.is_numeric_dtype(d[c])]
    for c in ("flag_iforest", "flag_zscore", "flag_iqr", "is_anomaly"):
        d[c] = False
    d["anomaly_score"], d["votes"] = 0.0, 0
    if not feats:
        return d
    X = d[feats].fillna(d[feats].median())
    if IsolationForest is not None:
        m = IsolationForest(contamination=sensitivity, random_state=0).fit(X)
        d["anomaly_score"] = -m.score_samples(X)
        d["flag_iforest"] = m.predict(X) == -1
    z = ((X - X.mean()) / X.std(ddof=0).replace(0, 1)).abs()
    d["flag_zscore"] = (z > z_thresh).any(axis=1)
    q1, q3 = X.quantile(.25), X.quantile(.75)
    iqr = q3 - q1
    d["flag_iqr"] = ((X < q1 - iqr_factor * iqr) | (X > q3 + iqr_factor * iqr)).any(axis=1)
    d["votes"] = d[["flag_iforest", "flag_zscore", "flag_iqr"]].sum(axis=1)
    d["is_anomaly"] = d["votes"] >= 2
    return d


# ---------- charts ----------
def gauge(score):
    color = "#2e7d32" if score >= 90 else "#f9a825" if score >= 70 else "#c62828"
    f = go.Figure(go.Indicator(mode="gauge+number", value=score, number={"suffix": "/100"},
                               gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}}))
    f.update_layout(height=230, margin=dict(t=20, b=0, l=20, r=20))
    return f


def dimension_bar(dims):
    f = go.Figure(go.Bar(x=list(dims.values()), y=[k.title() for k in dims], orientation="h",
                         marker_color="#1976d2", text=list(dims.values()), textposition="auto"))
    f.update_layout(title="Quality by dimension", xaxis_range=[0, 100], height=260, margin=dict(t=40, b=10))
    return f


def error_rate_bar(rates):
    f = go.Figure(go.Bar(x=list(rates.values()), y=list(rates.keys()), orientation="h", marker_color="#e53935"))
    f.update_layout(title="Affected rows by incident (%)", height=300, margin=dict(t=40, b=10, l=10),
                    yaxis=dict(autorange="reversed"))
    return f


def revenue_chart(imp):
    f = go.Figure(go.Waterfall(
        x=["Corrected revenue", "Duplicates", "Bad totals", "Other", "Reported revenue"],
        y=[imp["corrected"], imp["from_duplicates"], imp["from_bad_totals"], imp["other"], 0],
        measure=["absolute", "relative", "relative", "relative", "total"]))
    f.update_layout(title="How errors inflate reported revenue (INR)", height=340, margin=dict(t=40, b=10))
    return f


def anomaly_scatter(d):
    f = go.Figure()
    for flag, name, col in [(False, "Normal", "#90a4ae"), (True, "Anomaly", "#d32f2f")]:
        s = d[d["is_anomaly"] == flag]
        f.add_trace(go.Scatter(x=s.index, y=s["order_total"], mode="markers", name=name,
                               marker=dict(color=col, size=5 if not flag else 9)))
    f.update_layout(title="Order totals - anomalies highlighted", xaxis_title="row", yaxis_title="order_total",
                    yaxis_type="log", height=340, margin=dict(t=40, b=10))
    return f
