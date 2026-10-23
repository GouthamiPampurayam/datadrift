"""DataDrift - Streamlit dashboard. Run: python -m streamlit run app.py"""
import json
import pandas as pd
import streamlit as st
import analysis as an
import pipeline as pl
from ai_explainer import DEFAULT_MODEL, explain_incident, ollama_available

st.set_page_config(page_title="DataDrift", page_icon="🛡️", layout="wide")
SEV = {"high": "🔴", "medium": "🟠", "low": "🟡"}
PAGES = ["🏠 Overview", "🔍 Data Explorer", "🚨 Incidents", "📊 Anomalies", "🤖 AI Insights", "🔧 Fix Studio"]

# ================= sidebar =================
st.sidebar.markdown("## 🛡️ DataDrift")
st.sidebar.caption("AI Pipeline Guardian")
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

st.sidebar.markdown("**Data Source**")
src = st.sidebar.radio("Data Source", ["📦 Demo Dataset", "⬆️ Upload File"], label_visibility="collapsed")
inject = True
if src == "📦 Demo Dataset":
    inject = st.sidebar.checkbox("Inject data issues", True, help="Simulate a bad day in the pipeline")

st.sidebar.markdown("**Analysis Settings**")
sens = st.sidebar.slider("Anomaly sensitivity", 0.01, 0.20, 0.05, 0.01)
zthr = st.sidebar.slider("Z-score threshold σ", 1.5, 5.0, 3.0, 0.1)
iqrf = st.sidebar.slider("IQR fence factor", 1.0, 3.0, 1.5, 0.1)

st.sidebar.markdown("**AI Backend**")
backend = st.sidebar.selectbox("AI Backend", ["Ollama", "Rule-based (no LLM)"], label_visibility="collapsed")
use_llm = backend == "Ollama"
model = DEFAULT_MODEL
if use_llm:
    model = st.sidebar.text_input("Model", DEFAULT_MODEL)
    if ollama_available():
        st.sidebar.success("Ollama connected")
    else:
        st.sidebar.warning("Ollama offline - rule-based fallback will be used")

# ================= data =================
if src == "⬆️ Upload File":
    f = st.sidebar.file_uploader("CSV file (order schema)", type="csv")
    if f is None:
        st.title("DataDrift")
        st.info("Upload a CSV in the sidebar, or switch to the Demo Dataset.")
        st.stop()
    raw = pd.read_csv(f)
else:
    clean_demo = pl.make_clean_data()
    raw = pl.inject_issues(clean_demo) if inject else clean_demo

ukey = (src, inject, len(raw), tuple(raw.columns))
if st.session_state.get("ukey") != ukey:
    st.session_state.ukey, st.session_state.ai = ukey, {}

incidents = pl.detect_incidents(raw)
clean, fixlog = pl.clean_data(raw, incidents)
score = an.quality_score(incidents, len(raw))
impact = an.revenue_impact(raw, clean, incidents)
an_df = an.detect_anomalies(raw, sens, zthr, iqrf)


def run_ai(inc):
    st.session_state.ai[inc["id"]] = explain_incident(inc, model, use_llm)


def ai_card(a):
    st.write(a["explanation"])
    st.markdown(f"**Likely cause:** {a['likely_cause']}")
    st.markdown(f"**Confidence:** `{a['confidence']}`  ·  **Evidence used:** {', '.join(a['evidence_used']) or '—'}")
    st.success(f"**Suggested fix:** {a['suggested_fix']}")
    st.caption(("✅ safe to auto-fix" if a["safe_to_autofix"] else "⚠️ needs human review") + f" · {a['source']}")


# ================= pages =================
if page == PAGES[0]:
    st.title("🏠 Overview")
    st.caption("Detect data-quality issues, see their business impact, and get evidence-grounded AI explanations.")
    if score["overall"] >= 98 and not incidents:
        st.success("Pipeline healthy - no data-quality issues detected.")
    else:
        st.error(f"Pipeline needs attention - {len(incidents)} incidents detected.")
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.2])
    c1.plotly_chart(an.gauge(score["overall"]), use_container_width=True)
    c2.metric("Rows ingested", f"{len(raw):,}")
    c3.metric("Incidents", len(incidents))
    if impact:
        c4.metric("Revenue overstated by", f"₹{impact['difference']:,.0f}", f"{impact['difference_pct']}%", delta_color="inverse")
    a, b = st.columns(2)
    a.plotly_chart(an.dimension_bar(score["dimensions"]), use_container_width=True)
    if incidents:
        b.plotly_chart(an.error_rate_bar(an.error_rates(incidents, len(raw))), use_container_width=True)
    if impact and incidents:
        st.plotly_chart(an.revenue_chart(impact), use_container_width=True)

elif page == PAGES[1]:
    st.title("🔍 Data Explorer")
    st.subheader("Schema check vs expected contract")
    rows = []
    for col, kind in pl.EXPECTED_SCHEMA.items():
        present = col in raw.columns
        rows.append({"column": col, "expected": kind, "present": "✅" if present else "❌ missing",
                     "actual dtype": str(raw[col].dtype) if present else "-"})
    for col in raw.columns:
        if col not in pl.EXPECTED_SCHEMA:
            rows.append({"column": col, "expected": "-", "present": "🆕 unexpected", "actual dtype": str(raw[col].dtype)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Column profile")
    prof = pd.DataFrame({"dtype": raw.dtypes.astype(str), "nulls": raw.isna().sum(),
                         "null %": (100 * raw.isna().mean()).round(2), "unique": raw.nunique()})
    st.dataframe(prof, use_container_width=True)
    st.subheader("Raw data")
    mode = st.radio("Show", ["All rows", "Duplicate rows", "Rows with nulls"], horizontal=True)
    view = raw
    if mode == "Duplicate rows":
        view = raw[raw.duplicated(keep=False)]
    elif mode == "Rows with nulls":
        view = raw[raw.isna().any(axis=1)]
    st.caption(f"{len(view):,} rows")
    st.dataframe(view.head(500), use_container_width=True)

elif page == PAGES[2]:
    st.title("🚨 Incidents")
    if not incidents:
        st.success("No incidents detected.")
    for i in incidents:
        with st.expander(f"{SEV[i['severity']]} {i['id']} · {i['title']} · {i['affected_rows']:,} rows"):
            st.markdown(f"**Type:** `{i['type']}` · **Severity:** {i['severity']} · **Column:** {i['column']}")
            st.json(i["evidence"])

elif page == PAGES[3]:
    st.title("📊 Anomalies")
    n_an = int(an_df["is_anomaly"].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Anomalies (≥2 of 3 agree)", n_an)
    m2.metric("Isolation Forest", int(an_df["flag_iforest"].sum()))
    m3.metric("Z-score", int(an_df["flag_zscore"].sum()))
    m4.metric("IQR fence", int(an_df["flag_iqr"].sum()))
    st.plotly_chart(an.anomaly_scatter(an_df), use_container_width=True)
    st.caption("Three detectors vote; tune them with the sidebar sliders. Rows flagged by at least two are reported.")
    cols = [c for c in ["order_id", "product", "quantity", "unit_price", "order_total", "votes", "anomaly_score"] if c in an_df]
    st.dataframe(an_df[an_df["is_anomaly"]].sort_values("anomaly_score", ascending=False)[cols].head(100), use_container_width=True)

elif page == PAGES[4]:
    st.title("🤖 AI Insights")
    st.caption("Each explanation is generated from the incident's evidence only. Cited evidence keys are verified.")
    if st.button("Explain all incidents", type="primary"):
        with st.spinner("Analysing with the model..."):
            for i in incidents:
                run_ai(i)
    for i in incidents:
        with st.expander(f"{SEV[i['severity']]} {i['id']} · {i['title']}", expanded=i["id"] in st.session_state.ai):
            if st.button("Explain this incident", key="b" + i["id"]):
                with st.spinner("Thinking..."):
                    run_ai(i)
            a = st.session_state.ai.get(i["id"])
            if a:
                ai_card(a)
            else:
                st.caption("No explanation yet.")

else:
    st.title("🔧 Fix Studio")
    st.subheader("Safe fixes applied")
    for line in fixlog or ["Nothing to fix."]:
        st.write("✅ " + line)
    after_inc = pl.detect_incidents(clean.drop(columns=["needs_review"], errors="ignore"))
    after = an.quality_score(after_inc, len(clean))
    b1, b2, b3 = st.columns(3)
    b1.metric("Quality score", f"{after['overall']}", f"{after['overall'] - score['overall']:+.1f} vs before")
    b2.metric("Rows", f"{len(clean):,}", f"{len(clean) - len(raw):+,}")
    if impact:
        b3.metric("Revenue (corrected)", f"₹{impact['corrected']:,.0f}", f"{-impact['difference']:,.0f}", delta_color="off")
    if "needs_review" in clean:
        st.subheader("Quarantined for human review")
        st.dataframe(clean[clean["needs_review"]].head(100), use_container_width=True)
    st.subheader("Cleaned data preview")
    st.dataframe(clean.head(100), use_container_width=True)
    report = {"quality_score_before": score, "quality_score_after": after, "revenue_impact": impact,
              "incidents": incidents, "ai_explanations": st.session_state.ai, "fixes_applied": fixlog}
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Cleaned CSV", clean.to_csv(index=False), "clean_orders.csv", "text/csv")
    d2.download_button("⬇️ Incident report (JSON)", json.dumps(report, indent=2, default=str), "incident_report.json")
