import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from pathlib import Path

# ── Data Load ─────────────────────────────────────────────
RAW = Path("data/raw")
wb  = pd.read_csv(RAW / "wb_pakistan_health_indicators.csv")
who = pd.read_csv(RAW / "who_pakistan_catastrophic_exp.csv")
ml  = pd.read_csv(RAW / "ml_ready_dataset.csv")

wb  = wb.sort_values("year").dropna(subset=["oop_pct_current_health_exp"])
who_clean = who.groupby("year")[
    ["catastrophic_10pct","catastrophic_25pct"]].mean().reset_index()

# ── App ───────────────────────────────────────────────────
app = Dash(__name__)
app.title = "Pakistan CHE Dashboard"

COLORS = {
    "danger"  : "#E8593C",
    "safe"    : "#3B6D11",
    "blue"    : "#185FA5",
    "amber"   : "#BA7517",
    "bg"      : "#F8F9FA",
    "card"    : "#FFFFFF",
    "text"    : "#2C2C2A",
    "muted"   : "#6B6B67",
}

def card(children, style=None):
    base = {
        "background"  : COLORS["card"],
        "borderRadius": "12px",
        "padding"     : "20px 24px",
        "boxShadow"   : "0 1px 4px rgba(0,0,0,0.08)",
        "border"      : "0.5px solid #E0DED6",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)

def kpi(label, value, color=COLORS["text"]):
    return html.Div([
        html.P(label, style={"fontSize":"12px","color":COLORS["muted"],
                             "margin":"0 0 4px"}),
        html.P(value, style={"fontSize":"26px","fontWeight":"500",
                             "color":color,"margin":"0"}),
    ], style={"background":COLORS["bg"],"borderRadius":"8px",
              "padding":"14px 18px"})

# ── Layout ────────────────────────────────────────────────
app.layout = html.Div(style={
    "fontFamily":"'Segoe UI', sans-serif",
    "background" : COLORS["bg"],
    "minHeight"  : "100vh",
    "padding"    : "24px 32px",
}, children=[

    # Header
    html.Div([
        html.H1("Pakistan Catastrophic Health Expenditure",
                style={"fontSize":"22px","fontWeight":"500",
                       "color":COLORS["text"],"margin":"0"}),
        html.P("ML-powered analysis | WHO SDG 3.8 | World Bank Data",
               style={"fontSize":"13px","color":COLORS["muted"],"margin":"4px 0 0"}),
    ], style={"marginBottom":"24px"}),

    # KPI Row
    html.Div([
        kpi("Latest OOP %",        "52.9%",  COLORS["danger"]),
        kpi("WHO Threshold",       "40.0%",  COLORS["amber"]),
        kpi("Years Above Threshold","24/24", COLORS["danger"]),
        kpi("Catastrophic Rate",   "93.8%",  COLORS["danger"]),
        kpi("Model AUC (RF)",      "0.803",  COLORS["blue"]),
        kpi("Dataset Size",        "55,500", COLORS["safe"]),
    ], style={"display":"grid",
              "gridTemplateColumns":"repeat(6,1fr)",
              "gap":"12px","marginBottom":"20px"}),

    # Row 1 — OOP Trend + WHO Catastrophic
    html.Div([
        card([
            html.H3("OOP Expenditure Trend",
                    style={"fontSize":"14px","fontWeight":"500","margin":"0 0 12px"}),
            dcc.Graph(id="oop-trend", config={"displayModeBar":False},
                      style={"height":"280px"}),
        ], {"flex":"1"}),
        card([
            html.H3("Catastrophic Expenditure — WHO",
                    style={"fontSize":"14px","fontWeight":"500","margin":"0 0 12px"}),
            dcc.Graph(id="who-trend", config={"displayModeBar":False},
                      style={"height":"280px"}),
        ], {"flex":"1"}),
    ], style={"display":"flex","gap":"16px","marginBottom":"16px"}),

    # Row 2 — Income Filter + Risk Distribution
    html.Div([
        card([
            html.H3("Risk by Income Group",
                    style={"fontSize":"14px","fontWeight":"500","margin":"0 0 8px"}),
            html.P("Filter income range (USD/month):",
                   style={"fontSize":"12px","color":COLORS["muted"],"margin":"0 0 8px"}),
            dcc.RangeSlider(
                id="income-slider", min=150, max=5000,
                value=[150, 5000], step=50,
                marks={150:"$150", 700:"$700",
                       1200:"$1,200", 2500:"$2,500", 5000:"$5,000"},
            ),
            dcc.Graph(id="risk-income", config={"displayModeBar":False},
                      style={"height":"240px","marginTop":"12px"}),
        ], {"flex":"1"}),

        card([
            html.H3("Severity Distribution",
                    style={"fontSize":"14px","fontWeight":"500","margin":"0 0 12px"}),
            dcc.Graph(id="severity-pie", config={"displayModeBar":False},
                      style={"height":"280px"}),
        ], {"flex":"1"}),
    ], style={"display":"flex","gap":"16px","marginBottom":"16px"}),

    # Row 3 — SHAP + Admission
    html.Div([
        card([
            html.H3("SHAP Feature Importance",
                    style={"fontSize":"14px","fontWeight":"500","margin":"0 0 12px"}),
            dcc.Graph(id="shap-bar", config={"displayModeBar":False},
                      style={"height":"280px"}),
        ], {"flex":"1"}),
        card([
            html.H3("Billing by Medical Condition",
                    style={"fontSize":"14px","fontWeight":"500","margin":"0 0 12px"}),
            dcc.Graph(id="billing-bar", config={"displayModeBar":False},
                      style={"height":"280px"}),
        ], {"flex":"1"}),
    ], style={"display":"flex","gap":"16px","marginBottom":"16px"}),

    # Footer
    html.P(
        "Data: World Bank | WHO GHO | Kaggle Healthcare Dataset  ·  "
        "Model: XGBoost + SHAP  ·  By Asif Nawaz, PMAS Arid Agriculture University",
        style={"fontSize":"11px","color":COLORS["muted"],
               "textAlign":"center","marginTop":"8px"}
    ),
])

# ── Callbacks ─────────────────────────────────────────────
@app.callback(Output("oop-trend","figure"), Input("income-slider","value"))
def oop_trend(_):
    fig = go.Figure()
    fig.add_hline(y=40, line_dash="dash",
                  line_color="gray", annotation_text="40% WHO threshold")
    fig.add_trace(go.Scatter(
        x=wb["year"], y=wb["oop_pct_current_health_exp"],
        mode="lines+markers", name="OOP %",
        line=dict(color=COLORS["danger"], width=2.5),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(232,89,60,0.08)"
    ))
    fig.update_layout(**_layout())
    fig.update_yaxes(title="OOP % of Current Health Exp")
    return fig

@app.callback(Output("who-trend","figure"), Input("income-slider","value"))
def who_trend(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=who_clean["year"], y=who_clean["catastrophic_10pct"],
        mode="lines+markers", name="10% threshold",
        line=dict(color=COLORS["danger"], width=2.5),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=who_clean["year"], y=who_clean["catastrophic_25pct"],
        mode="lines+markers", name="25% threshold",
        line=dict(color=COLORS["blue"], width=2.5),
        marker=dict(size=6),
    ))
    fig.update_layout(**_layout())
    fig.update_yaxes(title="Population (%)")
    return fig

@app.callback(Output("risk-income","figure"),
              Input("income-slider","value"))
def risk_income(income_range):
    filtered = ml[
        (ml["monthly_income_usd"] >= income_range[0]) &
        (ml["monthly_income_usd"] <= income_range[1])
    ]
    grouped = filtered.groupby("monthly_income_usd")["is_catastrophic"].mean().reset_index()
    grouped.columns = ["income","catastrophic_rate"]
    grouped = grouped.sort_values("income")

    fig = px.bar(grouped, x="income", y="catastrophic_rate",
                 color="catastrophic_rate",
                 color_continuous_scale=["#3B6D11","#BA7517","#E8593C"])
    fig.update_layout(**_layout())
    fig.update_yaxes(title="Catastrophic Rate", tickformat=".0%")
    fig.update_xaxes(title="Monthly Income (USD)")
    fig.update_coloraxes(showscale=False)
    return fig

@app.callback(Output("severity-pie","figure"), Input("income-slider","value"))
def severity_pie(income_range):
    filtered = ml[
        (ml["monthly_income_usd"] >= income_range[0]) &
        (ml["monthly_income_usd"] <= income_range[1])
    ]
    sev = filtered["severity"].value_counts().reset_index()
    sev.columns = ["severity","count"]
    colors = {
        "safe"                : COLORS["safe"],
        "catastrophic_mild"   : COLORS["amber"],
        "catastrophic_severe" : COLORS["danger"],
        "catastrophic_crisis" : "#7A1A08",
    }
    fig = px.pie(sev, names="severity", values="count",
                 color="severity",
                 color_discrete_map=colors,
                 hole=0.45)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(**_layout())
    return fig

@app.callback(Output("shap-bar","figure"), Input("income-slider","value"))
def shap_bar(_):
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
                            columns=["feature","shap"])
    df_shap = df_shap.sort_values("shap")
    colors  = [COLORS["danger"] if v > df_shap["shap"].median()
               else COLORS["blue"] for v in df_shap["shap"]]
    fig = go.Figure(go.Bar(
        x=df_shap["shap"], y=df_shap["feature"],
        orientation="h", marker_color=colors
    ))
    fig.update_layout(**_layout())
    fig.update_xaxes(title="mean |SHAP value|")
    return fig

@app.callback(Output("billing-bar","figure"), Input("income-slider","value"))
def billing_bar(_):
    avg = ml.groupby("Medical Condition")["Billing Amount"].mean(
        ).sort_values(ascending=True).reset_index()
    fig = go.Figure(go.Bar(
        x=avg["Billing Amount"], y=avg["Medical Condition"],
        orientation="h", marker_color=COLORS["blue"]
    ))
    fig.update_layout(**_layout())
    fig.update_xaxes(title="Avg Billing (USD)")
    return fig

def _layout():
    return dict(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(family="Segoe UI", size=11, color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="#EEEEE8", gridwidth=0.5),
        yaxis=dict(showgrid=True, gridcolor="#EEEEE8", gridwidth=0.5),
    )

server = app.server  

if __name__ == "__main__":
    app.run(debug=True)