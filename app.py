import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(
    page_title="FraudGuard",
    layout="centered"
)

st.markdown("""
<style>
[data-testid="stHeader"]           { display: none !important; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
.block-container { padding-top: 1rem; max-width: 720px; }
#MainMenu, footer { visibility: hidden; }

.fg-header {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    border-radius: 12px;
    padding: 20px 28px;
    color: white;
    margin-bottom: 24px;
}
.fg-header h1 { font-size: 22px; margin: 0; letter-spacing: 1px; }
.fg-header p  { font-size: 12px; margin: 4px 0 0 0; opacity: 0.6; }

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
# ONBOARDING HTML
# ─────────────────────────────────────────
ONBOARDING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f1f5f9;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:1.5rem 1rem;min-height:100vh}
.phone-wrap{background:#e2e8f0;border-radius:16px;padding:1.25rem;display:flex;justify-content:center;margin-bottom:1rem}
.phone{width:288px;background:#fff;border-radius:24px;border:1px solid #cbd5e1;overflow:hidden}
.phone-bar{background:#0f172a;padding:10px 16px;display:flex;align-items:center;gap:8px}
.phone-bar-dot{width:6px;height:6px;border-radius:50%;background:#94a3b8;opacity:.5}
.phone-bar-title{color:#e2e8f0;font-size:13px;font-weight:500;letter-spacing:.5px}
.phone-body{padding:18px 14px;min-height:410px;display:flex;flex-direction:column}
.step-pill{display:flex;gap:4px;margin-bottom:14px;justify-content:center}
.step-pip{width:20px;height:4px;border-radius:2px;background:#e2e8f0;transition:background .3s}
.step-pip.done{background:#0f172a}
.step-pip.active{background:#378ADD}
.ob-icon{width:46px;height:46px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:21px;margin:0 auto 10px}
.ob-title{font-size:15px;font-weight:500;color:#0f172a;text-align:center;margin-bottom:5px}
.ob-sub{font-size:12px;color:#64748b;text-align:center;line-height:1.6;margin-bottom:14px}
.ob-card{background:#f8fafc;border-radius:8px;padding:9px 11px;margin-bottom:7px;display:flex;align-items:center;gap:9px;border:1px solid #e2e8f0}
.ob-card-icon{font-size:15px;color:#64748b}
.ob-card-text{font-size:12px;color:#0f172a;line-height:1.4}
.ob-card-sub{font-size:11px;color:#64748b}
input.ob-input{width:100%;padding:7px 9px;border-radius:8px;border:1px solid #cbd5e1;background:#fff;font-size:12px;color:#0f172a;font-family:inherit;margin-bottom:7px}
input.ob-input:focus{outline:none;border-color:#378ADD}
.tag-row{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px}
.tag{padding:4px 9px;border-radius:20px;font-size:11px;border:1px solid #cbd5e1;cursor:pointer;color:#64748b;background:#fff;transition:all .2s}
.tag.sel{background:#E6F1FB;color:#0C447C;border-color:#85B7EB}
.ob-btn{width:100%;padding:9px;border-radius:8px;border:none;background:#0f172a;color:#f8fafc;font-size:12px;font-weight:500;cursor:pointer;margin-top:auto;font-family:inherit}
.ob-btn:hover{background:#1e3a5f}
.ob-btn.sec{background:transparent;color:#64748b;border:1px solid #cbd5e1;margin-top:5px}
.ob-btn.sec:hover{background:#f8fafc}
.check-row{display:flex;align-items:center;gap:9px;padding:7px 0;border-bottom:1px solid #f1f5f9}
.check-row:last-child{border-bottom:none}
.check-circle{width:18px;height:18px;border-radius:50%;border:1px solid #cbd5e1;display:flex;align-items:center;justify-content:center;font-size:10px;flex-shrink:0}
.check-circle.done{background:#EAF3DE;border-color:#639922;color:#3B6D11}
.nav-row{display:flex;justify-content:space-between;align-items:center;width:100%;max-width:316px}
.nav-label{font-size:11px;color:#64748b}
.nav-btns{display:flex;gap:5px}
.nav-b{background:#fff;border:1px solid #cbd5e1;border-radius:8px;padding:5px 12px;cursor:pointer;font-size:11px;color:#0f172a;font-family:inherit}
.nav-b:hover{background:#f8fafc}
.nav-b:disabled{opacity:.35;cursor:default}
</style>
</head>
<body>
<div class="phone-wrap">
  <div class="phone">
    <div class="phone-bar">
      <div class="phone-bar-dot"></div>
      <div class="phone-bar-title">FraudGuard</div>
    </div>
    <div class="phone-body" id="phone-body"></div>
  </div>
</div>
<div class="nav-row">
  <span class="nav-label" id="nav-label">Step 1 of 5</span>
  <div class="nav-btns">
    <button class="nav-b" id="prev-b" onclick="go(-1)" disabled>← Back</button>
    <button class="nav-b" id="next-b" onclick="go(1)">Next →</button>
  </div>
</div>
<script>
const STEPS=[
  {render:()=>`
    <div class="step-pill">${pips(0)}</div>
    <div class="ob-icon" style="background:#E6F1FB"><i class="ti ti-shield-check" style="font-size:21px;color:#185FA5"></i></div>
    <div class="ob-title">Welcome to FraudGuard</div>
    <div class="ob-sub">Real-time credit card fraud detection powered by machine learning. Let's set you up in under a minute.</div>
    <div class="ob-card"><i class="ti ti-bolt ob-card-icon"></i><div><div class="ob-card-text">Instant scoring</div><div class="ob-card-sub">Each transaction scored in &lt;1 second</div></div></div>
    <div class="ob-card"><i class="ti ti-chart-bar ob-card-icon"></i><div><div class="ob-card-text">XGBoost + SVM hybrid model</div><div class="ob-card-sub">97% AUC-ROC · 91% F1 score</div></div></div>
    <div class="ob-card"><i class="ti ti-history ob-card-icon"></i><div><div class="ob-card-text">Full transaction history</div><div class="ob-card-sub">Track approvals and blocks over time</div></div></div>
    <button class="ob-btn" onclick="go(1)">Get started</button>
  `},
  {render:()=>`
    <div class="step-pill">${pips(1)}</div>
    <div class="ob-icon" style="background:#EAF3DE"><i class="ti ti-user" style="font-size:21px;color:#3B6D11"></i></div>
    <div class="ob-title">Your profile</div>
    <div class="ob-sub">This helps us personalise your fraud risk baseline.</div>
    <input class="ob-input" placeholder="Full name" value="Levi Omondi"/>
    <input class="ob-input" placeholder="Email address" value="levi@example.com"/>
    <input class="ob-input" placeholder="Phone number" value="+254 712 345 678"/>
    <div style="font-size:10px;color:#64748b;margin-bottom:10px">Your data is used only for fraud scoring and is never sold.</div>
    <button class="ob-btn" onclick="go(1)">Continue</button>
    <button class="ob-btn sec" onclick="go(1)">Skip for now</button>
  `},
  {render:()=>`
    <div class="step-pill">${pips(2)}</div>
    <div class="ob-icon" style="background:#FAEEDA"><i class="ti ti-building-store" style="font-size:21px;color:#854F0B"></i></div>
    <div class="ob-title">Your usual merchants</div>
    <div class="ob-sub">Select where you shop regularly to reduce false positives.</div>
    <div class="tag-row">
      ${['Naivas Supermarket','Quickmart','Java House','KFC Kenya','Jumia Kenya','Netflix','Uber Kenya','KPLC Prepaid','Carrefour Kenya','Glovo Kenya']
        .map((m,i)=>`<span class="tag${[0,1,3,5,6].includes(i)?' sel':''}" onclick="this.classList.toggle('sel')">${m}</span>`).join('')}
    </div>
    <div style="font-size:10px;color:#64748b;margin-bottom:10px">Trusted merchants are weighted lower in the risk model.</div>
    <button class="ob-btn" onclick="go(1)">Save merchants</button>
  `},
  {render:()=>`
    <div class="step-pill">${pips(3)}</div>
    <div class="ob-icon" style="background:#EEEDFE"><i class="ti ti-bell" style="font-size:21px;color:#534AB7"></i></div>
    <div class="ob-title">Alert preferences</div>
    <div class="ob-sub">Choose when FraudGuard should notify you.</div>
    <div style="margin-bottom:10px">
      ${[
        {icon:'ti-device-mobile',label:'Push notification',sub:'Instant alert on your phone',sel:true},
        {icon:'ti-mail',label:'Email alert',sub:'Sent to levi@example.com',sel:true},
        {icon:'ti-message',label:'SMS alert',sub:'Sent to +254 712 345 678',sel:false},
      ].map(a=>`
        <div class="check-row">
          <div class="check-circle${a.sel?' done':''}">${a.sel?'<i class="ti ti-check" style="font-size:9px"></i>':''}</div>
          <i class="ti ${a.icon}" style="font-size:14px;color:#64748b"></i>
          <div><div style="font-size:12px;color:#0f172a">${a.label}</div><div style="font-size:10px;color:#64748b">${a.sub}</div></div>
        </div>`).join('')}
    </div>
    <div style="font-size:11px;color:#64748b;margin-bottom:5px">Alert threshold</div>
    <div class="tag-row">
      ${['High risk only','Medium + High','All flags'].map((t,i)=>`<span class="tag${i===0?' sel':''}" onclick="document.querySelectorAll('.tag-row:last-of-type .tag').forEach(x=>x.classList.remove('sel'));this.classList.add('sel')">${t}</span>`).join('')}
    </div>
    <button class="ob-btn" onclick="go(1)">Save preferences</button>
  `},
  {render:()=>`
    <div class="step-pill">${pips(4)}</div>
    <div class="ob-icon" style="background:#E6F1FB"><i class="ti ti-circle-check" style="font-size:21px;color:#185FA5"></i></div>
    <div class="ob-title">You're all set</div>
    <div class="ob-sub">Here's a summary of your FraudGuard setup.</div>
    <div style="margin-bottom:10px">
      ${[
        {label:'Profile',val:'Levi Omondi'},
        {label:'Trusted merchants',val:'5 saved'},
        {label:'Alerts',val:'Push + Email'},
        {label:'Alert threshold',val:'High risk only'},
        {label:'Model',val:'XGBoost · threshold 50%'},
      ].map(r=>`
        <div class="check-row">
          <div class="check-circle done"><i class="ti ti-check" style="font-size:9px"></i></div>
          <div style="flex:1;display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:12px;color:#64748b">${r.label}</span>
            <span style="font-size:12px;font-weight:500;color:#0f172a">${r.val}</span>
          </div>
        </div>`).join('')}
    </div>
    <button class="ob-btn" onclick="window.parent.postMessage({type:'onboarding_complete'},'*')">Go to dashboard</button>
  `}
];
let cur=0;
function pips(active){return STEPS.map((_,i)=>`<div class="step-pip${i<active?' done':i===active?' active':''}"></div>`).join('')}
function render(){
  document.getElementById('phone-body').innerHTML=STEPS[cur].render();
  document.getElementById('nav-label').textContent=`Step ${cur+1} of ${STEPS.length}`;
  document.getElementById('prev-b').disabled=cur===0;
  document.getElementById('next-b').disabled=cur===STEPS.length-1;
}
function go(d){cur=Math.max(0,Math.min(STEPS.length-1,cur+d));render()}
render();
</script>
</body>
</html>
"""

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
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)

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

DEFAULT_AGE      = 35
DEFAULT_GENDER   = "M"
DEFAULT_CITY     = "Columbia"
DEFAULT_STATE    = "NY"
DEFAULT_DISTANCE = 5.0

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if 'onboarded' not in st.session_state:
    st.session_state.onboarded = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'result' not in st.session_state:
    st.session_state.result = None

# ─────────────────────────────────────────
# ONBOARDING SCREEN
# ─────────────────────────────────────────
if not st.session_state.onboarded:
    components.html(ONBOARDING_HTML, height=620, scrolling=False)

    col_skip, col_done = st.columns([3, 1])
    with col_skip:
        if st.button("Skip onboarding", use_container_width=True):
            st.session_state.onboarded = True
            st.rerun()
    with col_done:
        if st.button("Done ✓", use_container_width=True):
            st.session_state.onboarded = True
            st.rerun()
    st.stop()

# ─────────────────────────────────────────
# MAIN APP (shown after onboarding)
# ─────────────────────────────────────────
if not model_loaded:
    st.error(f"Could not load model files: {model_error}")
    st.info("Make sure fraud_model.pkl, label_encoders.pkl, feature_names.pkl and threshold.pkl are in the same folder as app.py")
    st.stop()

# ─────────────────────────────────────────
# FRAUD CHECK
# ─────────────────────────────────────────
def run_fraud_check(merchant, amount_kes, hour_override=None):
    now        = datetime.now()
    cat_raw, merch_raw = MERCHANTS[merchant]
    job_raw    = "Chartered accountant"
    use_hour   = hour_override if hour_override is not None else now.hour
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
    return reasons


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
