import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from datetime import datetime

st.set_page_config(
    page_title="FraudGuard",
    layout="centered"
)

st.markdown("""
<style>
[data-testid="stHeader"]        { display: none !important; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
.block-container { padding-top: 1rem; max-width: 720px; }
#MainMenu, footer               { visibility: hidden; }

.fg-header {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    border-radius: 12px;
    padding: 20px 28px;
    color: white;
    margin-bottom: 24px;
}
.fg-header h1  { font-size: 22px; margin: 0; letter-spacing: 1px; }
.fg-header p   { font-size: 12px; margin: 4px 0 0 0; opacity: 0.6; }

.card-safe {
    background: #f0fdf4;
    border: 1.5px solid #16a34a;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.card-fraud {
    background: #fff1f2;
    border: 1.5px solid #dc2626;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.card-safe  h2 { color: #15803d; margin: 8px 0 4px 0; }
.card-fraud h2 { color: #b91c1c; margin: 8px 0 4px 0; }

.risk-track {
    background: #e5e7eb;
    border-radius: 99px;
    height: 10px;
    margin: 10px 0 4px 0;
    overflow: hidden;
}
.risk-fill { height: 10px; border-radius: 99px; transition: width 0.4s ease; }

.fg-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 4px;
}

.reason-tag {
    background: #fff1f2;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    color: #b91c1c;
    margin: 4px 0;
    display: block;
}

.hist-row {
    background: white;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #16a34a;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
}
.hist-row.blocked { border-left-color: #dc2626; }

.stButton > button {
    background: #0f172a;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px;
    width: 100%;
}
.stButton > button:hover { background: #1e3a5f; color: white; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load('fraud_model.pkl')
    encoders  = joblib.load('label_encoders.pkl')
    features  = joblib.load('feature_names.pkl')
    threshold = joblib.load('threshold.pkl')
    return model, encoders, features, float(threshold)

try:
    model, encoders, feature_names, threshold = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info("Make sure fraud_model.pkl, label_encoders.pkl, feature_names.pkl and threshold.pkl are in the same folder as app.py")
    st.stop()

# ─────────────────────────────────────────
# MAPPINGS
# ─────────────────────────────────────────
MERCHANTS = {
    "Naivas Supermarket":   ("grocery_pos",   "fraud_Connelly-Carter"),
    "Quickmart":            ("grocery_pos",   "fraud_Conroy Ltd"),
    "Carrefour Kenya":      ("shopping_pos",  "fraud_Cormier LLC"),
    "Jumia Kenya":          ("shopping_net",  "fraud_Cummings Group"),
    "Netflix":              ("misc_net",      "fraud_Cummings LLC"),
    "Showmax":              ("misc_net",      "fraud_Corwin-Collins"),
    "Java House":           ("food_dining",   "fraud_Crooks and Sons"),
    "KFC Kenya":            ("food_dining",   "fraud_Crist, Jakubowski and Littel"),
    "Glovo Kenya":          ("grocery_net",   "fraud_Conroy-Emard"),
    "Uber Kenya":           ("gas_transport", "fraud_Conroy-Cruickshank"),
    "KPLC Prepaid":         ("misc_pos",      "fraud_Corwin-Gorczany"),
    "Total Energies":       ("gas_transport", "fraud_Corwin-Romaguera"),
    "Goodlife Pharmacy":    ("health_fitness","fraud_Cremin, Hamill and Reichel"),
    "SGR Madaraka Express": ("travel",        "fraud_Cronin, Kshlerin and Weber"),
    "Nairobi Cinema":       ("entertainment", "fraud_Crona and Sons"),
}

# ─────────────────────────────────────────
# DEFAULTS for all hidden fields
# ─────────────────────────────────────────
DEFAULT_AGE        = 35
DEFAULT_GENDER     = "M"
DEFAULT_CITY       = "Columbia"
DEFAULT_STATE      = "NY"
DEFAULT_DISTANCE   = 5.0

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'result' not in st.session_state:
    st.session_state.result = None

# ─────────────────────────────────────────
# FRAUD CHECK
# ─────────────────────────────────────────
def run_fraud_check(merchant, amount_kes, hour_override=None):
    now        = datetime.now()
    cat_raw, merch_raw = MERCHANTS[merchant]
    job_raw    = "Chartered accountant"
    use_hour   = hour_override if hour_override is not None else random.randint(0, 23)
    amount_usd = float(amount_kes) / 130

    row = {
        'merchant':    encoders['merchant'].transform([merch_raw])[0],
        'category':    encoders['category'].transform([cat_raw])[0],
        'amt':         amount_usd,
        'gender':      encoders['gender'].transform([DEFAULT_GENDER])[0],
        'city':        encoders['city'].transform([DEFAULT_CITY])[0],
        'state':       encoders['state'].transform([DEFAULT_STATE])[0],
        'job':         encoders['job'].transform([job_raw])[0],
        'hour':        use_hour,
        'day':         now.day,
        'month':       now.month,
        'dayofweek':   now.weekday(),
        'age':         DEFAULT_AGE,
        'distance_km': DEFAULT_DISTANCE,
    }

    df_in    = pd.DataFrame([row])[feature_names]
    prob     = float(model.predict_proba(df_in)[0][1])
    is_fraud = prob >= threshold
    return prob, is_fraud, cat_raw, use_hour

def fraud_reasons(merchant, amount_kes, hour):
    reasons = []
    cat = MERCHANTS[merchant][0]
    if amount_kes / 130 > 200:
        reasons.append("Unusually high transaction amount")
    if cat in ('shopping_net', 'misc_net', 'grocery_net'):
        reasons.append("High-risk online merchant category")
    if hour >= 22 or hour <= 3:
        reasons.append("Transaction during high-risk hours (10 PM – 4 AM)")
    return reasons

# ─────────────────────────────────────────
# RISK BAR
# ─────────────────────────────────────────
def render_risk_bar(prob):
    pct   = int(prob * 100)
    color = "#16a34a" if prob < 0.4 else "#f59e0b" if prob < threshold else "#dc2626"
    label = "Low" if prob < 0.4 else "Medium" if prob < threshold else "High"
    st.markdown(f"""
        <div class="fg-label">Fraud Risk — {label} ({pct}%)</div>
        <div class="risk-track">
            <div class="risk-fill" style="width:{pct}%; background:{color};"></div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:11px;color:#94a3b8;margin-bottom:12px;">
            <span>0%</span>
            <span>Threshold {int(threshold*100)}%</span>
            <span>100%</span>
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
    <div class="fg-header">
        <h1>FraudGuard</h1>
        <p>Real-time credit card fraud detection · XGBoost · BSc Data Science · JKUAT</p>
    </div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab_check, tab_history = st.tabs(["Check Transaction", "History"])

# ══════════════════════════════════════════
# TAB 1 — CHECK TRANSACTION
# ══════════════════════════════════════════
with tab_check:

    if st.session_state.result:
        r    = st.session_state.result
        hour = r['hour']

        if not r['is_fraud']:
            st.markdown(f"""
                <div class="card-safe">
                    <h2>Transaction Approved</h2>
                    <div style="font-size:15px;color:#374151;">
                        <b>{r['merchant']}</b> &nbsp;·&nbsp; KES {r['amount']:,.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="card-fraud">
                    <h2>Transaction Blocked</h2>
                    <div style="font-size:15px;color:#374151;">
                        <b>{r['merchant']}</b> &nbsp;·&nbsp; KES {r['amount']:,.2f}
                    </div>
                    <div style="font-size:13px;color:#b91c1c;margin-top:6px;">
                        No amount was deducted from your account.
                    </div>
                </div>
            """, unsafe_allow_html=True)

            reasons = fraud_reasons(r['merchant'], r['amount'], hour)
            if reasons:
                st.markdown("**Why was this flagged?**")
                for reason in reasons:
                    st.markdown(
                        f'<span class="reason-tag">{reason}</span>',
                        unsafe_allow_html=True
                    )

        render_risk_bar(r['prob'])

        if st.button("Check Another Transaction"):
            st.session_state.result = None
            st.rerun()

    else:
        st.markdown("#### Transaction Details")

        col1, col2 = st.columns(2)
        with col1:
            merchant = st.selectbox("Merchant", list(MERCHANTS))

        with col2:
            amount = st.number_input(
                "Amount (KES)", min_value=1.0,
                max_value=500000.0, value=1500.0, step=100.0
            )

        with st.expander("Demo Controls (presentation use only)"):
            demo_on  = st.checkbox("Override transaction hour")
            sim_hour = st.slider("Hour of day", 0, 23, 23, disabled=not demo_on)

        st.markdown("---")

        if st.button("Check for Fraud", use_container_width=True):
            with st.spinner("Analysing transaction..."):
                hour_val = sim_hour if demo_on else None
                prob, is_fraud, cat, use_hour = run_fraud_check(merchant, amount, hour_val)

            result = {
                'merchant':  merchant,
                'amount':    amount,
                'prob':      prob,
                'is_fraud':  is_fraud,
                'hour':      use_hour,
                'timestamp': datetime.now().strftime("%d %b %Y, %H:%M"),
                'category':  cat,
            }
            st.session_state.result = result
            st.session_state.history.insert(0, result)
            st.rerun()

        st.markdown(
            f"<div style='text-align:center;font-size:11px;color:#94a3b8;"
            f"margin-top:16px;'>Model: XGBoost · "
            f"PR-AUC 0.934 · Threshold {int(threshold*100)}% · "
            f"Fraud Recall 0.87</div>",
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════
with tab_history:
    if not st.session_state.history:
        st.info("No transactions checked yet. Use the Check Transaction tab.")
    else:
        total   = len(st.session_state.history)
        blocked = sum(1 for r in st.session_state.history if r['is_fraud'])
        safe    = total - blocked

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Checked", total)
        c2.metric("Approved",      safe)
        c3.metric("Blocked",       blocked)

        st.markdown("---")

        for r in st.session_state.history:
            css   = "hist-row blocked" if r['is_fraud'] else "hist-row"
            badge = "Blocked" if r['is_fraud'] else "Approved"
            st.markdown(f"""
                <div class="{css}">
                    <div style="display:flex;justify-content:space-between;">
                        <span><b>{r['merchant']}</b> &nbsp;·&nbsp; {badge}</span>
                        <span style="font-weight:600;">KES {r['amount']:,.2f}</span>
                    </div>
                    <div style="color:#64748b;margin-top:3px;">
                        Risk: {r['prob']*100:.1f}% &nbsp;·&nbsp; {r['timestamp']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()