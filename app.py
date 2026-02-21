"""
Credit Risk Analyzer — Streamlit Dark Dashboard
"""
import sys, os
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from src.predict import load_model, predict
from src.features import add_feature

MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ── Must be the FIRST Streamlit call ─────────────────────────────────────────
st.set_page_config(
    page_title="CreditRisk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load model once (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

MODEL, SCALER = load_artifacts()
EXPECTED_COLS = list(SCALER.feature_names_in_)

# ── Dark-theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ─── palette ─── */
:root {
  --bg-base   : #0D1117;
  --bg-surface: #161B22;
  --bg-card   : #1C2333;
  --bg-input  : #1C2333;
  --border    : #30363D;
  --border-hi : #58A6FF;
  --text-hi   : #E6EDF3;
  --text-mid  : #8B949E;
  --text-low  : #484F58;
  --accent    : #58A6FF;
  --accent2   : #BC8CFF;
  --green     : #3FB950;
  --orange    : #F78166;
  --yellow    : #E3B341;
  --radius-lg : 16px;
  --radius-md : 12px;
  --radius-sm : 8px;
  --shadow    : 0 4px 24px rgba(0,0,0,0.5);
}

/* ─── base ─── */
html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif !important;
  background-color: var(--bg-base) !important;
  color: var(--text-hi) !important;
}
.block-container {
  padding: 20px 28px 32px !important;
  max-width: 1440px !important;
  background: var(--bg-base) !important;
}

/* ─── inputs ─── */
input, textarea {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-hi) !important;
  font-size: 14px !important;
}
input:focus, textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}
/* number inputs */
[data-testid="stNumberInput"] input {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-hi) !important;
  border-radius: var(--radius-sm) !important;
}
/* text input */
[data-testid="stTextInput"] input {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-hi) !important;
}
/* selectbox */
[data-baseweb="select"] > div {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-hi) !important;
}
[data-baseweb="popover"] [role="option"] {
  background: var(--bg-card) !important;
  color: var(--text-hi) !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
  background: rgba(88,166,255,0.15) !important;
  color: var(--accent) !important;
}
/* slider */
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; }
.stSlider > div > div { background: var(--border) !important; }
.stSlider > div > div > div { background: var(--accent) !important; }

/* ─── labels & captions ─── */
label, .stMarkdown p, .stCaption {
  color: var(--text-mid) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
}

/* ─── button ─── */
div.stButton > button {
  background: linear-gradient(135deg, #58A6FF, #BC8CFF) !important;
  color: #0D1117 !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: 14px 32px !important;
  font-size: 14px !important;
  letter-spacing: 0.6px !important;
  text-transform: uppercase !important;
  box-shadow: 0 4px 20px rgba(88,166,255,0.35) !important;
  width: 100% !important;
  transition: all 0.25s !important;
}
div.stButton > button:hover {
  background: linear-gradient(135deg, #79BBFF, #D2AAFF) !important;
  box-shadow: 0 8px 28px rgba(88,166,255,0.5) !important;
  transform: translateY(-2px) !important;
}

/* ─── tabs ─── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  color: var(--text-mid) !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 10px 20px !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(88,166,255,0.1) !important;
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}

/* ─── spinner ─── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ─── scrollbar ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ─── hide Streamlit chrome ─── */
#MainMenu, footer, header { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers — all cards use dark palette
# ─────────────────────────────────────────────────────────────────────────────

CARD = (
    'background:#1C2333;border-radius:16px;padding:{pad};'
    'border:1px solid #30363D;box-shadow:0 4px 24px rgba(0,0,0,0.5);'
)

def stat_card(icon, label, value, sub, sub_color="#3FB950"):
    return f"""
<div style="{CARD.format(pad='18px 20px')}">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
    <div style="width:38px;height:38px;border-radius:10px;background:#0D1117;
                display:flex;align-items:center;justify-content:center;font-size:17px;">{icon}</div>
    <span style="font-size:10px;font-weight:700;color:#8B949E;text-transform:uppercase;letter-spacing:.8px;">{label}</span>
  </div>
  <div style="font-size:22px;font-weight:800;color:#E6EDF3;">{value}</div>
  <div style="font-size:11px;font-weight:600;color:{sub_color};margin-top:4px;">{sub}</div>
</div>"""


def bar_chart_html(values, color_grad, title, subtitle, icon):
    labels = ["Jan","Feb","Mar","Apr","May","Jun"]
    mx = max(values) if max(values) > 0 else 1
    bars = ""
    for i, v in enumerate(values):
        h = max(6, int((v / mx) * 110))
        bars += f"""
<div style="display:flex;flex-direction:column;align-items:center;gap:3px;flex:1;">
  <span style="font-size:9px;color:#8B949E;font-weight:600;">${v/1000:.0f}k</span>
  <div style="width:100%;height:{h}px;background:{color_grad};border-radius:5px 5px 3px 3px;"></div>
  <span style="font-size:9px;color:#484F58;">{labels[i]}</span>
</div>"""
    return f"""
<div style="{CARD.format(pad='20px')}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
    <div>
      <div style="font-size:14px;font-weight:700;color:#E6EDF3;">{title}</div>
      <div style="font-size:10px;color:#8B949E;margin-top:2px;">{subtitle}</div>
    </div>
    <div style="width:30px;height:30px;border-radius:8px;background:#0D1117;
                display:flex;align-items:center;justify-content:center;font-size:13px;">{icon}</div>
  </div>
  <div style="display:flex;align-items:flex-end;gap:6px;height:130px;padding-top:10px;">{bars}</div>
</div>"""


def donut_html(pct, risk_color, risk_label):
    deg = int(pct * 3.6)
    bg  = f"conic-gradient({risk_color} 0deg,{risk_color} {deg}deg,#30363D {deg}deg,#30363D 360deg)"
    return f"""
<div style="{CARD.format(pad='22px')};display:flex;flex-direction:column;align-items:center;justify-content:center;">
  <div style="font-size:14px;font-weight:700;color:#E6EDF3;margin-bottom:2px;">Risk Score</div>
  <div style="font-size:10px;color:#8B949E;margin-bottom:18px;">Default Probability</div>
  <div style="width:140px;height:140px;border-radius:50%;background:{bg};
              display:flex;align-items:center;justify-content:center;">
    <div style="width:104px;height:104px;border-radius:50%;background:#1C2333;
                display:flex;flex-direction:column;align-items:center;justify-content:center;">
      <div style="font-size:28px;font-weight:900;color:{risk_color};">{pct:.1f}</div>
      <div style="font-size:10px;color:#8B949E;font-weight:600;">percent</div>
    </div>
  </div>
  <div style="margin-top:14px;padding:6px 18px;background:rgba(0,0,0,0.3);
              border:1px solid {risk_color}55;border-radius:50px;
              font-size:11px;font-weight:700;color:{risk_color};letter-spacing:1px;">{risk_label}</div>
</div>"""


def client_profile_html(name, age, ps, avg_pay):
    return f"""
<div style="{CARD.format(pad='20px')}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
    <div>
      <div style="font-size:14px;font-weight:700;color:#E6EDF3;">Client Profile</div>
      <div style="font-size:10px;color:#8B949E;margin-top:2px;">{name}</div>
    </div>
    <div style="width:30px;height:30px;border-radius:8px;background:#0D1117;
                display:flex;align-items:center;justify-content:center;font-size:13px;">👤</div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
    <div style="background:#0D1117;border-radius:10px;padding:14px;text-align:center;border:1px solid #30363D;">
      <div style="font-size:20px;font-weight:800;color:#58A6FF;">{age}</div>
      <div style="font-size:9px;font-weight:700;color:#484F58;text-transform:uppercase;margin-top:3px;">Age</div>
    </div>
    <div style="background:#0D1117;border-radius:10px;padding:14px;text-align:center;border:1px solid #30363D;">
      <div style="font-size:20px;font-weight:800;color:#3FB950;">{ps}</div>
      <div style="font-size:9px;font-weight:700;color:#484F58;text-transform:uppercase;margin-top:3px;">Delay Mo.</div>
    </div>
    <div style="background:#0D1117;border-radius:10px;padding:14px;text-align:center;border:1px solid #30363D;">
      <div style="font-size:20px;font-weight:800;color:#BC8CFF;">6</div>
      <div style="font-size:9px;font-weight:700;color:#484F58;text-transform:uppercase;margin-top:3px;">Months</div>
    </div>
    <div style="background:#0D1117;border-radius:10px;padding:14px;text-align:center;border:1px solid #30363D;">
      <div style="font-size:20px;font-weight:800;color:#E3B341;">${avg_pay/1000:.1f}k</div>
      <div style="font-size:9px;font-weight:700;color:#484F58;text-transform:uppercase;margin-top:3px;">Avg Pay</div>
    </div>
  </div>
</div>"""


def insight_row(level, label, value, desc):
    color_map = {"critical":"#F78166","warning":"#E3B341","good":"#3FB950"}
    icon_map  = {"critical":"❌","warning":"⚠️","good":"✅"}
    c = color_map[level]; ic = icon_map[level]
    return f"""
<div style="display:flex;align-items:center;gap:12px;padding:12px 16px;
            background:#0D1117;border-radius:10px;border:1px solid {c}22;">
  <div style="font-size:16px;">{ic}</div>
  <div style="flex:1;">
    <div style="font-size:12px;font-weight:700;color:#E6EDF3;">{label}</div>
    <div style="font-size:10px;color:#8B949E;margin-top:1px;">{desc}</div>
  </div>
  <div style="font-size:14px;font-weight:800;color:{c};">{value}</div>
</div>"""


def progress_bar_html(pct, risk_color, risk_label, bar_color):
    bar_width = min(pct, 100)
    return f"""
<div style="{CARD.format(pad='20px')};margin-bottom:16px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
    <div style="font-size:14px;font-weight:700;color:#E6EDF3;">Default Probability Scale</div>
    <div style="padding:5px 14px;background:rgba(0,0,0,0.4);border:1px solid {risk_color}55;
                border-radius:50px;font-size:11px;font-weight:700;color:{risk_color};">{pct:.1f}%</div>
  </div>
  <div style="height:10px;background:#30363D;border-radius:8px;overflow:hidden;">
    <div style="height:100%;width:{bar_width:.0f}%;background:{bar_color};border-radius:8px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:6px;font-size:9px;color:#484F58;font-weight:600;">
    <span>0% Safe</span><span>30%</span><span>60%</span><span>100% Critical</span>
  </div>
</div>"""


def recommendation_html(risk_color, risk_label, risk_emoji, rec_text):
    return f"""
<div style="background:rgba(0,0,0,0.3);border:1px solid {risk_color}44;
            border-radius:16px;padding:20px;margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
    <div style="width:40px;height:40px;border-radius:10px;background:#1C2333;
                display:flex;align-items:center;justify-content:center;font-size:18px;
                border:1px solid {risk_color}44;">{risk_emoji}</div>
    <div>
      <div style="font-size:14px;font-weight:700;color:#E6EDF3;">Recommendation</div>
      <div style="font-size:10px;color:#8B949E;">AI Engine Assessment</div>
    </div>
  </div>
  <div style="font-size:13px;color:#C9D1D9;line-height:1.8;">{rec_text}</div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction + result builder
# ─────────────────────────────────────────────────────────────────────────────

def do_predict(name, age, gender, education, marital,
               credit_limit, bills, pays, pay_delay_months):
    sex_map = {"Male":1,"Female":2}
    edu_map = {"Post-Graduate":1,"University":2,"High School":3,"Others":4}
    mar_map = {"Single":1,"Married":2,"Others":3}

    ps = max(-1, min(int(pay_delay_months), 9))
    b1,b2,b3,b4,b5,b6 = bills
    p1,p2,p3,p4,p5,p6 = pays

    row = {
        "LIMIT_BAL":credit_limit,
        "SEX":sex_map.get(gender,1),
        "EDUCATION":edu_map.get(education,2),
        "MARRIAGE":mar_map.get(marital,2),
        "AGE":int(age),
        "PAY_0":ps,"PAY_2":ps,"PAY_3":ps,"PAY_4":ps,"PAY_5":ps,"PAY_6":ps,
        "BILL_AMT1":b1,"BILL_AMT2":b2,"BILL_AMT3":b3,
        "BILL_AMT4":b4,"BILL_AMT5":b5,"BILL_AMT6":b6,
        "PAY_AMT1":p1,"PAY_AMT2":p2,"PAY_AMT3":p3,
        "PAY_AMT4":p4,"PAY_AMT5":p5,"PAY_AMT6":p6,
    }
    df = pd.DataFrame([row])
    df = add_feature(df)
    df = pd.get_dummies(df, columns=["SEX","EDUCATION","MARRIAGE"], drop_first=False)
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = 0
    df = df[EXPECTED_COLS]
    X  = SCALER.transform(df)
    return predict(MODEL, X) * 100, ps


def build_result(name, age, gender, education, marital,
                 credit_limit, bills, pays, pay_delay_months):

    pct, ps = do_predict(name, age, gender, education, marital,
                         credit_limit, bills, pays, pay_delay_months)

    # ── risk tier ─────────────────────────────────────────────────────────────
    if pct < 30:
        risk_color = "#3FB950"; risk_label = "LOW RISK"
        risk_emoji = "✅"
        rec_text   = "Strong repayment profile. Client shows consistent financial discipline. <strong>Approval recommended</strong> with standard terms."
        bar_color  = "linear-gradient(90deg,#3FB950,#2EA043)"
    elif pct < 60:
        risk_color = "#E3B341"; risk_label = "MEDIUM RISK"
        risk_emoji = "⚠️"
        rec_text   = "Moderate risk indicators present. Irregular payment patterns. <strong>Additional review recommended</strong>."
        bar_color  = "linear-gradient(90deg,#E3B341,#BB8A00)"
    else:
        risk_color = "#F78166"; risk_label = "HIGH RISK"
        risk_emoji = "❌"
        rec_text   = "High probability of default. Significant repayment risk. <strong>Decline recommended</strong>."
        bar_color  = "linear-gradient(90deg,#F78166,#DA3633)"

    avg_bill    = float(np.mean(bills))
    avg_pay     = float(np.mean(pays))
    utilization = min(100, int(round(avg_bill / credit_limit * 100))) if credit_limit > 0 else 0
    pay_ratio   = min(100, int(round(avg_pay / (avg_bill + 1) * 100)))
    client_name = name.strip() if name and name.strip() else "Anonymous Client"

    util_color = "#F78166" if utilization > 50 else "#3FB950"
    util_label = "High"    if utilization > 50 else "Normal"
    pr_color   = "#F78166" if pay_ratio < 30 else "#3FB950"
    pr_label   = "Low"     if pay_ratio < 30 else "Healthy"

    # ── insights ───────────────────────────────────────────────────────────────
    ins = []
    ins.append(("critical" if utilization>80 else "warning" if utilization>50 else "good",
                "Credit Utilization", f"{utilization}%",
                "High utilization" if utilization>80 else "Above 30%" if utilization>50 else "Within healthy limits"))
    ins.append(("critical" if pay_ratio<20 else "warning" if pay_ratio<50 else "good",
                "Payment Ratio", f"{pay_ratio}%",
                "Very low" if pay_ratio<20 else "Partial repayments" if pay_ratio<50 else "Strong repayment"))
    ins.append(("critical" if ps>=3 else "warning" if ps>=1 else "good",
                "Payment Delays", f"{ps} mo" if ps>0 else "On Time",
                "Severe signal" if ps>=3 else "Minor delays" if ps>=1 else "No delays"))
    insights_html = "\n".join(insight_row(*i) for i in ins)

    # ── assemble ───────────────────────────────────────────────────────────────
    stats = f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px;">
  {stat_card("💳","Credit Limit",f"${credit_limit:,.0f}","Active")}
  {stat_card("📈","Utilization",f"{utilization}%",util_label,util_color)}
  {stat_card("💰","Avg Bill",f"${avg_bill:,.0f}","6-month avg","#8B949E")}
  {stat_card("💸","Pay Ratio",f"{pay_ratio}%",pr_label,pr_color)}
</div>"""

    row3 = f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px;">
  {bar_chart_html(bills,"linear-gradient(180deg,#58A6FF,#1F6FEB)","Billing History","6-month trend","📊")}
  {donut_html(pct, risk_color, risk_label)}
  {client_profile_html(client_name, int(age), ps, avg_pay)}
</div>"""

    row2 = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;">
  {bar_chart_html(pays,"linear-gradient(180deg,#3FB950,#2EA043)","Payment History","6-month payments","💵")}
  <div style="{CARD.format(pad='20px')}">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
      <div>
        <div style="font-size:14px;font-weight:700;color:#E6EDF3;">Risk Analysis</div>
        <div style="font-size:10px;color:#8B949E;margin-top:2px;">Key indicators</div>
      </div>
      <div style="width:30px;height:30px;border-radius:8px;background:#0D1117;
                  display:flex;align-items:center;justify-content:center;font-size:13px;">🔍</div>
    </div>
    <div style="display:flex;flex-direction:column;gap:8px;">{insights_html}</div>
  </div>
</div>"""

    prog = progress_bar_html(pct, risk_color, risk_label, bar_color)
    rec  = recommendation_html(risk_color, risk_label, risk_emoji, rec_text)
    foot = '<div style="text-align:center;padding:6px 0 2px;"><span style="font-size:10px;color:#484F58;">Analysis by CreditRisk AI v5.0 | Requires human review</span></div>'

    return stats + row3 + row2 + prog + rec + foot


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

# ── Header bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#161B22;border-radius:14px;padding:16px 24px;margin-bottom:20px;
            display:flex;align-items:center;justify-content:space-between;
            border:1px solid #30363D;box-shadow:0 4px 24px rgba(0,0,0,0.5);">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:44px;height:44px;border-radius:12px;
                background:linear-gradient(135deg,#58A6FF,#BC8CFF);
                display:flex;align-items:center;justify-content:center;
                font-size:18px;font-weight:800;color:#0D1117;
                box-shadow:0 4px 16px rgba(88,166,255,0.35);">CR</div>
    <div>
      <div style="font-size:20px;font-weight:800;color:#E6EDF3;">
        Credit<span style="color:#58A6FF;">Risk</span> Dashboard
      </div>
      <div style="font-size:11px;color:#8B949E;">AI-Powered Risk Intelligence Platform</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="display:flex;align-items:center;gap:7px;padding:7px 14px;
                background:rgba(63,185,80,0.1);border:1px solid rgba(63,185,80,0.3);
                border-radius:50px;">
      <div style="width:7px;height:7px;border-radius:50%;background:#3FB950;"></div>
      <span style="font-size:11px;font-weight:600;color:#3FB950;">System Online</span>
    </div>
    <div style="font-size:10px;font-weight:600;color:#8B949E;padding:5px 12px;
                background:#0D1117;border:1px solid #30363D;border-radius:6px;">v5.0</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main two-column area ──────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;
                padding-bottom:14px;border-bottom:1px solid #30363D;">
      <div style="width:38px;height:38px;border-radius:10px;background:#0D1117;
                  border:1px solid #30363D;display:flex;align-items:center;
                  justify-content:center;font-size:16px;">📋</div>
      <div>
        <div style="font-size:17px;font-weight:700;color:#E6EDF3;">Client Assessment</div>
        <div style="font-size:11px;color:#8B949E;">Enter applicant details for risk evaluation</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["👤  Demographics", "💳  Credit Profile", "📊  Financials"])

    with tab1:
        name_in = st.text_input("Client Full Name", placeholder="e.g. Sarah Johnson", label_visibility="visible")
        c1, c2 = st.columns(2)
        with c1:
            age_in = st.slider("Age", 18, 80, 32)
        with c2:
            gender_in = st.selectbox("Gender", ["Male", "Female"])
        c3, c4 = st.columns(2)
        with c3:
            education_in = st.selectbox("Education Level",
                ["Post-Graduate", "University", "High School", "Others"], index=1)
        with c4:
            marital_in = st.selectbox("Marital Status", ["Single", "Married", "Others"])

    with tab2:
        credit_in    = st.slider("Credit Limit ($)", 5_000, 1_000_000, 80_000, 5_000)
        pay_delay_in = st.slider("Payment Delay (months)", -1, 9, 0)
        st.caption("-1 = Paid duly  |  0 = No delay  |  1–9 = Months overdue")

    with tab3:
        st.markdown("**Billing History — 6 Months**")
        bc1, bc2 = st.columns(2)
        with bc1:
            b1 = st.number_input("Month 1 Bill ($)", value=10000, step=100)
            b2 = st.number_input("Month 2 Bill ($)", value=9500,  step=100)
            b3 = st.number_input("Month 3 Bill ($)", value=8800,  step=100)
        with bc2:
            b4 = st.number_input("Month 4 Bill ($)", value=9200, step=100)
            b5 = st.number_input("Month 5 Bill ($)", value=8500, step=100)
            b6 = st.number_input("Month 6 Bill ($)", value=9000, step=100)

        st.markdown("**Payment History — 6 Months**")
        pc1, pc2 = st.columns(2)
        with pc1:
            p1 = st.number_input("Month 1 Paid ($)", value=5000, step=100)
            p2 = st.number_input("Month 2 Paid ($)", value=4800, step=100)
            p3 = st.number_input("Month 3 Paid ($)", value=4500, step=100)
        with pc2:
            p4 = st.number_input("Month 4 Paid ($)", value=5200, step=100)
            p5 = st.number_input("Month 5 Paid ($)", value=4700, step=100)
            p6 = st.number_input("Month 6 Paid ($)", value=5000, step=100)

    analyze = st.button("🔍  Analyze Risk Profile", use_container_width=True)

with right:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;
                padding-bottom:14px;border-bottom:1px solid #30363D;">
      <div style="width:38px;height:38px;border-radius:10px;background:#0D1117;
                  border:1px solid #30363D;display:flex;align-items:center;
                  justify-content:center;font-size:16px;">📈</div>
      <div>
        <div style="font-size:17px;font-weight:700;color:#E6EDF3;">Risk Assessment</div>
        <div style="font-size:11px;color:#8B949E;">AI-generated analysis & insights</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    panel = st.empty()

    if not analyze:
        panel.markdown("""
<div style="text-align:center;padding:70px 24px;">
  <div style="width:80px;height:80px;border-radius:20px;background:#161B22;
              border:1px solid #30363D;display:flex;align-items:center;
              justify-content:center;margin:0 auto 18px;font-size:32px;">🏦</div>
  <div style="font-size:18px;font-weight:700;color:#E6EDF3;margin-bottom:8px;">Ready for Analysis</div>
  <div style="font-size:13px;color:#8B949E;max-width:280px;margin:0 auto;line-height:1.6;">
    Fill in client details and click
    <strong style="color:#58A6FF;">Analyze Risk Profile</strong>
    to generate a comprehensive risk report.
  </div>
</div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Running AI risk analysis…"):
            try:
                html = build_result(
                    name_in, age_in, gender_in, education_in, marital_in,
                    credit_in,
                    [b1, b2, b3, b4, b5, b6],
                    [p1, p2, p3, p4, p5, p6],
                    pay_delay_in,
                )
                panel.markdown(html, unsafe_allow_html=True)
            except Exception as e:
                panel.error(f"Prediction error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:14px 22px;margin-top:20px;background:#161B22;border-radius:12px;
            border:1px solid #30363D;">
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="width:24px;height:24px;border-radius:6px;
                background:linear-gradient(135deg,#58A6FF,#BC8CFF);
                display:flex;align-items:center;justify-content:center;
                font-size:10px;font-weight:800;color:#0D1117;">CR</div>
    <span style="font-size:11px;color:#8B949E;">CreditRisk Enterprise Engine</span>
  </div>
  <span style="font-size:10px;color:#484F58;">
    AI-generated assessments require human review before final decisions
  </span>
</div>
""", unsafe_allow_html=True)
