import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Pakistan CHE Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ── Data Load ─────────────────────────────────────────────
@st.cache_data
def load_data():
    RAW = Path("data/raw")
    wb  = pd.read_csv(RAW / "wb_pakistan_health_indicators.csv")
    who = pd.read_csv(RAW / "who_pakistan_catastrophic_exp.csv")
    ml  = pd.read_csv(RAW / "ml_ready_dataset.csv")
    wb  = wb.sort_values("year").dropna(subset=["oop_pct_current_health_exp"])
    who_clean = who.groupby("year")[
        ["catastrophic_10pct","catastrophic_25pct"]].mean().reset_index()
    return wb, who_clean, ml

wb, who_clean, ml = load_data()

# ── Header ────────────────────────────────────────────────
st.title("🏥 Pakistan Catastrophic Health Expenditure")
st.caption("ML-powered analysis | WHO SDG 3.8 | World Bank Data | By Asif Nawaz, PMAS Arid Agriculture University")
st.divider()

# ── KPIs ──────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Latest OOP %",        "52.9%",  "-4.5% vs 2022", delta_color="inverse")
k2.metric("WHO Threshold",       "40.0%",  "Danger line")
k3.metric("Years Above Threshold","24/24", "2000–2023",     delta_color="off")
k4.metric("Catastrophic Rate",   "93.8%",  "Simulated")
k5.metric("Model AUC (RF)",      "0.803",  "Leakage-free")
k6.metric("Dataset Size",        "55,500", "Records")

st.divider()

# ── Row 1 ─────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("OOP Expenditure Trend")
    fig = go.Figure()
    fig.add_hline(y=40, line_dash="dash", line_color="gray",
                  annotation_text="40% WHO threshold")
    fig.add_trace(go.Scatter(
        x=wb["year"], y=wb["oop_pct_current_health_exp"],
        mode="lines+markers", name="OOP %",
        line=dict(color="#E8593C", width=2.5),
        fill="tozeroy", fillcolor="rgba(232,89,60,0.08)"
    ))
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                      height=300, showlegend=False)
    fig.update_yaxes(title="OOP % of Current Health Exp")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Catastrophic Expenditure — WHO")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=who_clean["year"], y=who_clean["catastrophic_10pct"],
        mode="lines+markers", name="10% threshold",
        line=dict(color="#E8593C", width=2.5), marker=dict(size=6)
    ))
    fig2.add_trace(go.Scatter(
        x=who_clean["year"], y=who_clean["catastrophic_25pct"],
        mode="lines+markers", name="25% threshold",
        line=dict(color="#185FA5", width=2.5), marker=dict(size=6)
    ))
    fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=300)
    fig2.update_yaxes(title="Population (%)")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Income Filter ──────────────────────────────────────────
st.subheader("Interactive Analysis — Filter by Income")
income_range = st.slider(
    "Monthly Income Range (USD)",
    min_value=150, max_value=5000,
    value=(150, 5000), step=50,
    format="$%d"
)

filtered = ml[
    (ml["monthly_income_usd"] >= income_range[0]) &
    (ml["monthly_income_usd"] <= income_range[1])
]

col3, col4 = st.columns(2)

with col3:
    st.subheader("Risk by Income Group")
    grouped = filtered.groupby("monthly_income_usd")[
        "is_catastrophic"].mean().reset_index()
    grouped.columns = ["income","catastrophic_rate"]
    fig3 = px.bar(grouped, x="income", y="catastrophic_rate",
                  color="catastrophic_rate",
                  color_continuous_scale=["#3B6D11","#BA7517","#E8593C"])
    fig3.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                       height=300, coloraxis_showscale=False)
    fig3.update_yaxes(title="Catastrophic Rate", tickformat=".0%")
    fig3.update_xaxes(title="Monthly Income (USD)")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Severity Distribution")
    sev = filtered["severity"].value_counts().reset_index()
    sev.columns = ["severity","count"]
    colors = {
        "safe"                :"#3B6D11",
        "catastrophic_mild"   :"#BA7517",
        "catastrophic_severe" :"#E8593C",
        "catastrophic_crisis" :"#7A1A08",
    }
    fig4 = px.pie(sev, names="severity", values="count",
                  color="severity", color_discrete_map=colors,
                  hole=0.45)
    fig4.update_traces(textposition="outside", textinfo="percent+label")
    fig4.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=300)
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Row 3 ─────────────────────────────────────────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader("SHAP Feature Importance")
    shap_data = {
        "monthly_income_usd": 0.9534,
        "income_low"        : 0.2072,
        "Age"               : 0.1788,
        "length_of_stay"    : 0.1580,
        "insurer_enc"       : 0.0930,
        "condition_enc"     : 0.0891,
        "income_medium"     : 0.0812,
        "gender_enc"        : 0.0502,
        "is_emergency"      : 0.0441,
        "is_urgent"         : 0.0398,
        "is_chronic"        : 0.0334,
    }
    df_shap = pd.DataFrame(list(shap_data.items()),
                            columns=["feature","shap_value"])
    df_shap = df_shap.sort_values("shap_value")
    median_val = df_shap["shap_value"].median()
    df_shap["color"] = df_shap["shap_value"].apply(
        lambda x: "#E8593C" if x > median_val else "#185FA5")
    fig5 = go.Figure(go.Bar(
        x=df_shap["shap_value"], y=df_shap["feature"],
        orientation="h", marker_color=df_shap["color"]
    ))
    fig5.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=350)
    fig5.update_xaxes(title="mean |SHAP value|")
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.subheader("Billing by Medical Condition")
    avg = ml.groupby("Medical Condition")["Billing Amount"].mean(
        ).sort_values(ascending=True).reset_index()
    fig6 = go.Figure(go.Bar(
        x=avg["Billing Amount"], y=avg["Medical Condition"],
        orientation="h", marker_color="#185FA5"
    ))
    fig6.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=350)
    fig6.update_xaxes(title="Avg Billing (USD)")
    st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ── Footer ────────────────────────────────────────────────
st.caption(
    "Data: World Bank | WHO GHO | Kaggle Healthcare Dataset · "
    "Model: XGBoost + SHAP (AUC 0.803) · "
    "Asif Nawaz | PMAS Arid Agriculture University | MPhil Economics"
)