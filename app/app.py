"""
╔══════════════════════════════════════════════════════════════════╗
║  CREDIT RISK ANALYZER — Enterprise AI Engine                     ║
║  Premium Fintech Dashboard · Built with Gradio Blocks API        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys, os

# Path Setup (works regardless of CWD)
BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import joblib
import gradio as gr
from src.predict import load_model, predict
from src.features import add_feature

# Load ML Artefacts 
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

MODEL = load_model(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)
EXPECTED_COLS = list(SCALER.feature_names_in_)


#  PREDICTION ENGINE

def run_prediction(
    name, age, gender, education, marital,
    credit_limit, bill1, bill2, bill3, bill4, bill5, bill6,
    pay1, pay2, pay3, pay4, pay5, pay6,
    pay_delay_months
):
    sex_map  = {"Male": 1, "Female": 2}
    edu_map  = {"Post-Graduate": 1, "University": 2, "High School": 3, "Others": 4}
    mar_map  = {"Single": 1, "Married": 2, "Others": 3}

    ps = max(-1, min(int(pay_delay_months), 9))

    row = {
        "LIMIT_BAL": credit_limit,
        "SEX": sex_map.get(gender, 1),
        "EDUCATION": edu_map.get(education, 2),
        "MARRIAGE": mar_map.get(marital, 2),
        "AGE": int(age),
        "PAY_0": ps, "PAY_2": ps, "PAY_3": ps,
        "PAY_4": ps, "PAY_5": ps, "PAY_6": ps,
        "BILL_AMT1": bill1, "BILL_AMT2": bill2, "BILL_AMT3": bill3,
        "BILL_AMT4": bill4, "BILL_AMT5": bill5, "BILL_AMT6": bill6,
        "PAY_AMT1": pay1,  "PAY_AMT2": pay2,  "PAY_AMT3": pay3,
        "PAY_AMT4": pay4,  "PAY_AMT5": pay5,  "PAY_AMT6": pay6,
    }

    df = pd.DataFrame([row])
    df = add_feature(df)
    df = pd.get_dummies(df, columns=["SEX", "EDUCATION", "MARRIAGE"], drop_first=False)
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = 0
    df = df[EXPECTED_COLS]
    X = SCALER.transform(df)
    prob = predict(MODEL, X)
    pct = prob * 100

    # ── Risk classification ───
    if pct < 30:
        risk_color = "#10b981"
        risk_bg    = "rgba(16,185,129,0.08)"
        risk_border = "rgba(16,185,129,0.25)"
        risk_label = "LOW RISK"
        risk_icon  = "✅"
        risk_glow  = "0 0 40px rgba(16,185,129,0.3)"
        rec_text   = "Strong repayment profile detected. Client shows consistent financial discipline. <strong>Approval recommended</strong> with standard terms."
        bar_color  = "linear-gradient(90deg, #10b981, #34d399)"
    elif pct < 60:
        risk_color = "#f59e0b"
        risk_bg    = "rgba(245,158,11,0.08)"
        risk_border = "rgba(245,158,11,0.25)"
        risk_label = "MEDIUM RISK"
        risk_icon  = "⚠️"
        risk_glow  = "0 0 40px rgba(245,158,11,0.3)"
        rec_text   = "Moderate risk indicators present. Irregular payment patterns detected. <strong>Additional collateral or co-signer recommended</strong> before approval."
        bar_color  = "linear-gradient(90deg, #f59e0b, #fbbf24)"
    else:
        risk_color = "#ef4444"
        risk_bg    = "rgba(239,68,68,0.08)"
        risk_border = "rgba(239,68,68,0.25)"
        risk_label = "HIGH RISK"
        risk_icon  = "🚨"
        risk_glow  = "0 0 40px rgba(239,68,68,0.3)"
        rec_text   = "High probability of default. Significant repayment risk detected across multiple indicators. <strong>Decline recommended</strong>."
        bar_color  = "linear-gradient(90deg, #ef4444, #f87171)"

    # ── Derived metrics ───
    avg_bill = np.mean([bill1, bill2, bill3, bill4, bill5, bill6])
    avg_pay  = np.mean([pay1, pay2, pay3, pay4, pay5, pay6])
    utilization = min(100, int(round(avg_bill / credit_limit * 100))) if credit_limit > 0 else 0
    pay_ratio   = min(100, int(round(avg_pay / (avg_bill + 1) * 100)))

    client_display = name.strip() if name and name.strip() else "Client"

    # ── Build data-driven insight bullets ───
    insights = []

    # Credit utilization insight
    if utilization > 80:
        insights.append(f"🔴 <strong>High credit utilization ({utilization}%)</strong> — using most of available credit, a strong default predictor.")
    elif utilization > 50:
        insights.append(f"🟡 <strong>Moderate credit utilization ({utilization}%)</strong> — above the recommended 30% threshold.")
    else:
        insights.append(f"🟢 <strong>Healthy credit utilization ({utilization}%)</strong> — well within safe limits.")

    # Payment ratio insight
    if pay_ratio < 20:
        insights.append(f"🔴 <strong>Very low payment ratio ({pay_ratio}%)</strong> — client is paying back less than 20% of bills, indicating cash flow stress.")
    elif pay_ratio < 50:
        insights.append(f"🟡 <strong>Partial repayments ({pay_ratio}%)</strong> — client covers less than half of monthly bills on average.")
    else:
        insights.append(f"🟢 <strong>Strong payment ratio ({pay_ratio}%)</strong> — client consistently covers a significant portion of bills.")

    # Payment delay insight
    if ps >= 3:
        insights.append(f"🔴 <strong>Severe payment delays ({ps} months)</strong> — history of 3+ months overdue is the strongest default signal.")
    elif ps >= 1:
        insights.append(f"🟡 <strong>Minor payment delays ({ps} month{'s' if ps > 1 else ''})</strong> — some late payments detected in recent history.")
    elif ps == 0:
        insights.append("🟢 <strong>No payment delays</strong> — all recent payments made on time.")
    else:
        insights.append("🟢 <strong>Payments made ahead of schedule</strong> — client pays before due dates.")

    # Credit limit context
    if credit_limit < 30000:
        insights.append(f"🟡 <strong>Low credit limit (${credit_limit:,.0f})</strong> — limited credit history may increase uncertainty.")
    elif credit_limit > 200000:
        insights.append(f"🟢 <strong>High credit limit (${credit_limit:,.0f})</strong> — indicates strong banking relationship and trust.")

    # Bill trend insight
    bills = [bill1, bill2, bill3, bill4, bill5, bill6]
    if bills[0] > bills[-1] * 1.3:
        insights.append("🟡 <strong>Rising bill trend</strong> — recent bills are increasing, which may signal growing debt.")
    elif bills[-1] > bills[0] * 1.3:
        insights.append("🟢 <strong>Declining bill trend</strong> — bills are decreasing over time, suggesting debt reduction.")

    # Age context
    if int(age) < 25:
        insights.append(f"🟡 <strong>Young client (age {int(age)})</strong> — limited credit history increases risk uncertainty.")

    insights_html = "".join(f'<div class="insight-row">{i}</div>' for i in insights)

    # ═══ BUILD RESULT HTML ═══
    result_html = f"""
    <div class="result-container">

        <!-- Risk Hero -->
        <div class="risk-hero" style="border-color:{risk_border}; box-shadow:{risk_glow};">
            <div class="risk-badge" style="background:{risk_bg}; color:{risk_color}; border-color:{risk_border};">
                {risk_icon}&ensp;{risk_label}
            </div>
            <div class="risk-pct" style="color:{risk_color};">{pct:.1f}<span class="risk-pct-sign">%</span></div>
            <div class="risk-sub">DEFAULT PROBABILITY</div>
            <div class="risk-bar-track">
                <div class="risk-bar-fill" style="width:{min(pct, 100):.0f}%; background:{bar_color};"></div>
            </div>
            <div class="risk-client">{client_display}</div>
        </div>

        <!-- Metrics Grid -->
        <div class="metrics-grid">
            <div class="metric-tile">
                <div class="metric-icon">📊</div>
                <div class="metric-val">{utilization}%</div>
                <div class="metric-label">Credit Utilization</div>
            </div>
            <div class="metric-tile">
                <div class="metric-icon">💳</div>
                <div class="metric-val">${credit_limit:,.0f}</div>
                <div class="metric-label">Credit Limit</div>
            </div>
            <div class="metric-tile">
                <div class="metric-icon">🔄</div>
                <div class="metric-val">{pay_ratio}%</div>
                <div class="metric-label">Payment Ratio</div>
            </div>
            <div class="metric-tile">
                <div class="metric-icon">📈</div>
                <div class="metric-val">${avg_bill:,.0f}</div>
                <div class="metric-label">Avg Monthly Bill</div>
            </div>
        </div>

        <!-- Recommendation -->
        <div class="rec-box" style="border-left-color:{risk_color};">
            <div class="rec-title">
                <span style="color:{risk_color};">●</span>&ensp;Lending Recommendation
            </div>
            <div class="rec-body">{rec_text}</div>
        </div>

        <!-- Detailed Insights -->
        <div class="insights-box">
            <div class="insights-title">🔍&ensp;Why this rating?</div>
            {insights_html}
        </div>

    </div>
    """
    return result_html


#  THEME

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0b1120",
    body_text_color="#f1f5f9",
    block_background_fill="#111827",
    block_border_color="#1e293b",
    block_label_text_color="#e2e8f0",
    input_background_fill="#0f172a",
    input_border_color="#1e293b",
    button_primary_background_fill="linear-gradient(135deg, #059669, #10b981)",
    button_primary_text_color="#ffffff",
)



#  CUSTOM CSS

CSS = """
/* ─── GLOBAL ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

.gradio-container {
    background: linear-gradient(160deg, #0b1120 0%, #0f1729 40%, #0d1f2d 70%, #0b1120 100%) !important;
    max-width: 1380px !important;
    margin: 0 auto !important;
    padding: 24px 32px !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ─── HEADER ─── */
.app-header {
    background: linear-gradient(135deg, rgba(17,24,39,0.95), rgba(15,23,42,0.95));
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 16px;
    padding: 20px 32px;
    margin-bottom: 20px;
    backdrop-filter: blur(20px);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-left { display: flex; align-items: center; gap: 16px; }
.header-logo {
    width: 48px; height: 48px; border-radius: 14px;
    background: linear-gradient(135deg, #059669, #10b981);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; font-weight: 900; color: white;
    box-shadow: 0 0 24px rgba(16,185,129,0.3);
    flex-shrink: 0;
}
.header-title {
    font-size: 22px; font-weight: 800; color: #f1f5f9;
    letter-spacing: -0.3px; line-height: 1.2;
}
.header-title span { color: #10b981; }
.header-sub {
    font-size: 11px; color: #475569;
    letter-spacing: 2px; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace; margin-top: 2px;
}
.header-right { display: flex; align-items: center; gap: 20px; }
.header-status {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; color: #cbd5e1; font-weight: 500;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px rgba(16,185,129,0.6);
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 10px rgba(16,185,129,0.6); }
    50% { opacity: 0.7; box-shadow: 0 0 20px rgba(16,185,129,0.8); }
}
.header-ver {
    font-size: 11px; color: #334155;
    font-family: 'JetBrains Mono', monospace;
    padding: 4px 10px; border: 1px solid #1e293b;
    border-radius: 6px;
}

/* ─── CARDS / BLOCKS ─── */
.block {
    background: rgba(17,24,39,0.85) !important;
    border: 1px solid rgba(30,41,59,0.8) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    overflow: visible !important;
}
.block:hover {
    border-color: rgba(99,102,241,0.15) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
}

/* ─── SECTION HEADERS ─── */
.section-title {
    font-size: 13px; font-weight: 700; color: #cbd5e1;
    letter-spacing: 2px; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
    display: flex; align-items: center; gap: 8px;
}
.section-title::before {
    content: '';
    width: 3px; height: 14px;
    background: linear-gradient(180deg, #10b981, #059669);
    border-radius: 2px;
    display: inline-block;
}

/* ─── LABELS ─── */
label, label span, .label-wrap, .label-wrap span,
label > span, label > span > span,
.gradio-container label, .gradio-container label span,
.block label, .block label span {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: 0.5px !important;
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    border: none !important;
    box-shadow: none !important;
}

/* ─── INPUTS ─── */
input, textarea, select,
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #0f172a !important;
    background-color: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    -webkit-appearance: none !important;
    -moz-appearance: none !important;
    appearance: none !important;
    box-sizing: border-box !important;
}
input:focus, textarea:focus, select:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.12) !important;
    outline: none !important;
}
input::placeholder, textarea::placeholder {
    color: #334155 !important;
}

/* ─── DROPDOWNS ─── */
.wrap, .wrap-inner,
[data-testid="dropdown"],
.secondary-wrap,
.gradio-dropdown {
    background: #0f172a !important;
    background-color: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    -webkit-appearance: none !important;
    -moz-appearance: none !important;
    appearance: none !important;
}
.wrap input, .wrap-inner input,
[data-testid="dropdown"] input,
.secondary-wrap input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.wrap svg, .wrap-inner svg,
[data-testid="dropdown"] svg,
.secondary-wrap svg {
    color: #475569 !important;
}

/* Dropdown options list */
ul[role="listbox"], .dropdown-options {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}
ul[role="listbox"] li, .dropdown-options li {
    color: #e2e8f0 !important;
}
ul[role="listbox"] li:hover, .dropdown-options li:hover {
    background: rgba(16,185,129,0.1) !important;
}

/* ─── SLIDER ─── */
input[type="range"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.gradio-container span[data-testid="slider-number"] {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.2) !important;
    border-radius: 8px !important;
    color: #10b981 !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ─── TABS ─── */
.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid #1e293b !important;
    gap: 0 !important;
}
.tab-nav button {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
    margin-bottom: -1px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
    -webkit-appearance: none !important;
    appearance: none !important;
}
.tab-nav button:hover {
    color: #94a3b8 !important;
}
.tab-nav button.selected {
    color: #10b981 !important;
    border-bottom-color: #10b981 !important;
    font-weight: 700 !important;
    background: transparent !important;
    text-shadow: 0 0 20px rgba(16,185,129,0.3);
}

/* ─── PRIMARY BUTTON ─── */
button.primary {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 14px !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    box-shadow: 0 0 24px rgba(16,185,129,0.25), 0 4px 12px rgba(0,0,0,0.3) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    -webkit-appearance: none !important;
    appearance: none !important;
}
button.primary:hover {
    box-shadow: 0 0 40px rgba(16,185,129,0.35), 0 6px 20px rgba(0,0,0,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ─── MARKDOWN ─── */
.prose, .prose * {
    color: #e2e8f0 !important;
}
.prose h3, .prose h4 {
    color: #ffffff !important;
    font-weight: 700 !important;
}
.prose strong { color: #ffffff !important; }

/* ─── RESULT PANEL ─── */
.result-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.risk-hero {
    text-align: center;
    padding: 36px 28px 28px;
    background: linear-gradient(145deg, rgba(17,24,39,0.9), rgba(15,23,42,0.95));
    border: 1px solid;
    border-radius: 16px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.risk-hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(16,185,129,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.risk-badge {
    display: inline-block;
    font-size: 11px; font-weight: 800; letter-spacing: 3px;
    text-transform: uppercase;
    border: 1px solid; border-radius: 24px;
    padding: 6px 20px; margin-bottom: 18px;
    font-family: 'JetBrains Mono', monospace;
}
.risk-pct {
    font-size: 64px; font-weight: 900;
    line-height: 1; margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -2px;
}
.risk-pct-sign { font-size: 36px; opacity: 0.7; }
.risk-sub {
    font-size: 10px; font-weight: 700; color: #475569;
    letter-spacing: 3px; margin-bottom: 16px;
    font-family: 'JetBrains Mono', monospace;
}
.risk-bar-track {
    width: 80%; height: 4px; margin: 0 auto 14px;
    background: rgba(30,41,59,0.8); border-radius: 4px;
    overflow: hidden;
}
.risk-bar-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.6s ease;
}
.risk-client {
    font-size: 12px; color: #475569;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
}

.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 16px;
}
.metric-tile {
    background: rgba(15,23,42,0.7);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    transition: border-color 0.2s ease;
}
.metric-tile:hover { border-color: rgba(16,185,129,0.2); }
.metric-icon { font-size: 20px; margin-bottom: 8px; }
.metric-val {
    font-size: 22px; font-weight: 800; color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 10px; font-weight: 600; color: #475569;
    letter-spacing: 1.5px; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

.rec-box {
    background: rgba(15,23,42,0.7);
    border: 1px solid #1e293b;
    border-left: 3px solid;
    border-radius: 0 12px 12px 0;
    padding: 20px 22px;
}
.rec-title {
    font-size: 13px; font-weight: 700; color: #e2e8f0;
    margin-bottom: 8px; letter-spacing: 0.3px;
}
.rec-body {
    font-size: 13px; color: #64748b; line-height: 1.7;
}
.rec-body strong { color: #cbd5e1; }

.insights-box {
    background: rgba(15,23,42,0.7);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px 22px;
    margin-top: 12px;
}
.insights-title {
    font-size: 13px; font-weight: 700; color: #e2e8f0;
    margin-bottom: 12px; letter-spacing: 0.3px;
}
.insight-row {
    font-size: 12px; color: #cbd5e1; line-height: 1.6;
    padding: 8px 0;
    border-bottom: 1px solid rgba(30,41,59,0.5);
}
.insight-row:last-child { border-bottom: none; }
.insight-row strong { color: #f1f5f9; }

/* ─── FOOTER ─── */
.app-footer {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px; margin-top: 16px;
    font-size: 10px; color: #1e293b;
    letter-spacing: 1.5px; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    border-top: 1px solid rgba(30,41,59,0.5);
}
.app-footer span { color: #334155; }

/* ─── HIDE GRADIO FOOTER ─── */
footer { display: none !important; }

/* ─── NUMBER INPUT SPINNERS ─── */
input[type="number"] { -moz-appearance: textfield !important; }
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none !important; margin: 0 !important;
}

/* ─── PLACEHOLDER PANEL ─── */
.placeholder-panel {
    text-align: center; padding: 80px 24px;
    font-family: 'Inter', sans-serif;
}
.placeholder-icon {
    font-size: 48px; margin-bottom: 16px;
    opacity: 0.6;
}
.placeholder-text {
    font-size: 14px; color: #475569; font-weight: 500;
    line-height: 1.6;
}
.placeholder-text strong { color: #10b981; }
"""



#  BUILD UI
with gr.Blocks(title="CreditRisk Enterprise — AI Engine") as demo:

    # ── Header ──
    gr.HTML("""
    <div class="app-header">
        <div class="header-left">
            <div class="header-logo">CR</div>
            <div>
                <div class="header-title">CreditRisk <span>Enterprise</span></div>
                <div class="header-sub">AI-Powered Default Prediction Engine</div>
            </div>
        </div>
        <div class="header-right">
            <div class="header-status">
                <div class="status-dot"></div>
                <span>System Online</span>
            </div>
            <div class="header-ver">v5.0</div>
        </div>
    </div>
    """)

    with gr.Row():

        # ═══ LEFT PANEL — Inputs ═══
        with gr.Column(scale=5):

            gr.HTML('<div class="section-title">Client Assessment</div>')

            with gr.Tabs():

                # ── Tab 1: Demographics ──
                with gr.TabItem("👤 Demographics"):
                    name_in = gr.Textbox(
                        label="Client Full Name",
                        placeholder="e.g. Sarah Johnson",
                        max_lines=1,
                    )
                    with gr.Row():
                        age_in = gr.Slider(
                            minimum=18, maximum=80, value=32, step=1,
                            label="Age"
                        )
                        gender_in = gr.Dropdown(
                            choices=["Male", "Female"],
                            value="Male", label="Gender",
                            interactive=True,
                        )
                    with gr.Row():
                        education_in = gr.Dropdown(
                            choices=["Post-Graduate", "University", "High School", "Others"],
                            value="University", label="Education Level",
                            interactive=True,
                        )
                        marital_in = gr.Dropdown(
                            choices=["Single", "Married", "Others"],
                            value="Single", label="Marital Status",
                            interactive=True,
                        )

                # ── Tab 2: Credit Profile ──
                with gr.TabItem("💳 Credit Profile"):
                    credit_in = gr.Slider(
                        minimum=5000, maximum=1000000, value=80000, step=5000,
                        label="Credit Limit ($)"
                    )
                    pay_delay_in = gr.Slider(
                        minimum=-1, maximum=9, value=0, step=1,
                        label="Payment Delay Status (months)"
                    )
                    gr.Markdown(
                        "*-1 = Paid duly &nbsp;|&nbsp; 0 = No delay &nbsp;|&nbsp;"
                        " 1–9 = Months of delay*"
                    )

                # ── Tab 3: Financial History ──
                with gr.TabItem("📊 Financials"):
                    gr.Markdown("#### 6-Month Billing History")
                    with gr.Row():
                        with gr.Column():
                            b1 = gr.Number(value=10000, label="Month 1 Bill ($)")
                            b2 = gr.Number(value=9500,  label="Month 2 Bill ($)")
                            b3 = gr.Number(value=8800,  label="Month 3 Bill ($)")
                        with gr.Column():
                            b4 = gr.Number(value=9200,  label="Month 4 Bill ($)")
                            b5 = gr.Number(value=8500,  label="Month 5 Bill ($)")
                            b6 = gr.Number(value=9000,  label="Month 6 Bill ($)")

                    gr.Markdown("#### 6-Month Payment History")
                    with gr.Row():
                        with gr.Column():
                            p1 = gr.Number(value=5000, label="Month 1 Paid ($)")
                            p2 = gr.Number(value=4800, label="Month 2 Paid ($)")
                            p3 = gr.Number(value=4500, label="Month 3 Paid ($)")
                        with gr.Column():
                            p4 = gr.Number(value=5200, label="Month 4 Paid ($)")
                            p5 = gr.Number(value=4700, label="Month 5 Paid ($)")
                            p6 = gr.Number(value=5000, label="Month 6 Paid ($)")

            # ── Action Button ──
            btn = gr.Button(
                "⚡  Run Risk Analysis",
                variant="primary",
                size="lg",
            )

        # ═══ RIGHT PANEL — Results ═══
        with gr.Column(scale=4):
            gr.HTML('<div class="section-title">Risk Assessment</div>')
            pred_out = gr.HTML("""
            <div class="placeholder-panel">
                <div class="placeholder-icon">🏦</div>
                <div class="placeholder-text">
                    Enter client details and click<br>
                    <strong>Run Risk Analysis</strong> to begin
                </div>
            </div>
            """)

    # ── Footer ──
    gr.HTML("""
    <div class="app-footer">
        <span>CreditRisk Enterprise Engine · AI-Powered Analysis</span>
        <span>All assessments are model-generated and require human review</span>
    </div>
    """)

    # ── Wire up ──
    all_inputs = [
        name_in, age_in, gender_in, education_in, marital_in,
        credit_in,
        b1, b2, b3, b4, b5, b6,
        p1, p2, p3, p4, p5, p6,
        pay_delay_in,
    ]
    btn.click(fn=run_prediction, inputs=all_inputs, outputs=pred_out)

#  LAUNCH
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        theme=theme,
        css=CSS,
    )
