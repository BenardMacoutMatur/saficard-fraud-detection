import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from datetime import datetime
from database import (
    init_db, create_user, verify_pin, update_balance,
    name_exists, save_transaction, get_user_transactions,
    get_user_stats, get_all_transactions, get_all_users,
    get_system_stats, get_user_by_name
)

# ============================================================
# INIT DATABASE
# ============================================================
init_db()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SafiCard Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
    <style>
    .safi-header {
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 20px;
        letter-spacing: 1px;
    }
    .safi-tagline {
        font-size: 13px;
        opacity: 0.85;
        margin-top: 4px;
    }
    .credit-card {
        background: linear-gradient(135deg, #1B5E20, #388E3C);
        padding: 25px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-family: 'Courier New', monospace;
    }
    .txn-approved {
        background-color: #E8F5E9;
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #2E7D32;
        margin: 8px 0;
        color: #000;
    }
    .txn-blocked {
        background-color: #FFEBEE;
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #C62828;
        margin: 8px 0;
        color: #000;
    }
    .stButton>button {
        background-color: #1B5E20;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2E7D32;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_artifacts():
    model     = joblib.load('fraud_model.pkl')
    encoders  = joblib.load('label_encoders.pkl')
    features  = joblib.load('feature_names.pkl')
    threshold = joblib.load('threshold.pkl')
    return model, encoders, features, threshold

model, encoders, feature_names, threshold = load_artifacts()

# ============================================================
# MAPPINGS
# ============================================================
MERCHANTS = {
    "Naivas Supermarket":        ("grocery_pos",    "fraud_Connelly-Carter"),
    "Quickmart":                 ("grocery_pos",    "fraud_Conroy Ltd"),
    "Carrefour Kenya":           ("shopping_pos",   "fraud_Cormier LLC"),
    "Jumia Kenya":               ("shopping_net",   "fraud_Cummings Group"),
    "Netflix":                   ("misc_net",       "fraud_Cummings LLC"),
    "Showmax":                   ("misc_net",       "fraud_Corwin-Collins"),
    "Java House":                ("food_dining",    "fraud_Crooks and Sons"),
    "KFC Kenya":                 ("food_dining",    "fraud_Crist, Jakubowski and Littel"),
    "Glovo Kenya":               ("grocery_net",    "fraud_Conroy-Emard"),
    "Uber Kenya":                ("gas_transport",  "fraud_Conroy-Cruickshank"),
    "KPLC Prepaid":              ("misc_pos",       "fraud_Corwin-Gorczany"),
    "Total Energies":            ("gas_transport",  "fraud_Corwin-Romaguera"),
    "Goodlife Pharmacy":         ("health_fitness", "fraud_Cremin, Hamill and Reichel"),
    "SGR Madaraka Express":      ("travel",         "fraud_Cronin, Kshlerin and Weber"),
    "Nairobi Cinema":            ("entertainment",  "fraud_Crona and Sons"),
}

COUNTIES = {
    "Nairobi":  ("Columbia",   "NY"),
    "Mombasa":  ("Tampa",      "FL"),
    "Kisumu":   ("Houston",    "TX"),
    "Nakuru":   ("Sacramento", "CA"),
    "Eldoret":  ("Portland",   "WA"),
    "Thika":    ("Joliet",     "IL"),
    "Nyeri":    ("Cleveland",  "OH"),
    "Machakos": ("Omaha",      "NE"),
}

OCCUPATIONS = {
    "Software Developer":  "Applications developer",
    "Civil Engineer":      "Civil engineer, contracting",
    "Accountant":          "Chartered accountant",
    "Teacher":             "Secondary school teacher",
    "Doctor":              "Doctor, general practice",
    "Nurse":               "Mental health nurse",
    "Business Manager":    "Call centre manager",
    "Catering Manager":    "Catering manager",
    "Farmer":              "Agricultural consultant",
    "Chemical Engineer":   "Chemical engineer",
}

# ============================================================
# VALIDATE ENCODERS
# ============================================================
def validate_encoders():
    errors = []
    for merchant, (cat, merch_raw) in MERCHANTS.items():
        if merch_raw not in encoders['merchant'].classes_:
            errors.append(f"Merchant not found: {merch_raw}")
        if cat not in encoders['category'].classes_:
            errors.append(f"Category not found: {cat}")
    for county, (city, state) in COUNTIES.items():
        if city not in encoders['city'].classes_:
            errors.append(f"City not found: {city}")
        if state not in encoders['state'].classes_:
            errors.append(f"State not found: {state}")
    for occ, job_raw in OCCUPATIONS.items():
        if job_raw not in encoders['job'].classes_:
            errors.append(f"Job not found: {job_raw}")
    return errors

errors = validate_encoders()
if errors:
    st.error("Encoder mapping errors:")
    for e in errors:
        st.write(f"❌ {e}")
    st.stop()

# ============================================================
# SESSION STATE
# ============================================================
for key, val in {
    'logged_in':        False,
    'user':             None,
    'page':             'landing',
    'last_result':      None,
    'show_card':        False,
    'new_card_details': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================
# HELPERS
# ============================================================
def mask_card(card_number):
    try:
        parts = str(card_number).split(' ')
        if len(parts) == 4:
            return f"{parts[0]} **** **** {parts[3]}"
        return card_number
    except:
        return card_number

def refresh_user():
    if st.session_state.user:
        fresh = get_user_by_name(st.session_state.user['name'])
        if fresh:
            st.session_state.user.update({
                'balance':     float(fresh['balance']),
                'card_number': str(fresh['card_number']),
                'card_type':   str(fresh['card_type']),
                'expiry':      str(fresh['expiry']),
            })

# ============================================================
# FRAUD CHECK
# ============================================================
def check_fraud(user, merchant_name, amount_kes,
                sim_hour=None, amount_multiplier=1):
    now = datetime.now()
    category_raw, merchant_raw = MERCHANTS[merchant_name]
    city_raw, state_raw        = COUNTIES[user['county']]
    job_raw                    = OCCUPATIONS[user['occupation']]

    county_distances = {
        "Nairobi": 3.0,  "Mombasa": 8.0,  "Kisumu": 5.0,
        "Nakuru": 12.0,  "Eldoret": 15.0, "Thika": 6.0,
        "Nyeri": 20.0,   "Machakos": 10.0
    }
    distance   = county_distances.get(user['county'], 5.0) + random.uniform(0, 8)
    use_hour   = int(sim_hour) if sim_hour is not None else now.hour
    use_amount = (float(amount_kes) * float(amount_multiplier)) / 130

    encoded = {
        'merchant':    encoders['merchant'].transform([merchant_raw])[0],
        'category':    encoders['category'].transform([category_raw])[0],
        'amt':         use_amount,
        'gender':      encoders['gender'].transform([str(user['gender'])])[0],
        'city':        encoders['city'].transform([city_raw])[0],
        'state':       encoders['state'].transform([state_raw])[0],
        'job':         encoders['job'].transform([job_raw])[0],
        'hour':        use_hour,
        'day':         now.day,
        'month':       now.month,
        'dayofweek':   now.weekday(),
        'age':         int(user['age']),
        'distance_km': distance,
    }

    input_df = pd.DataFrame([encoded])[feature_names]
    prob     = model.predict_proba(input_df)[0][1]
    is_fraud = int(prob >= threshold)
    return prob, is_fraud, category_raw

# ============================================================
# PAGE: LANDING
# ============================================================
def page_landing():
    st.markdown('''
        <div class="safi-header">
            💳 SafiCard
            <div class="safi-tagline">Salama kila wakati — Safe every time</div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("### Kenya's Smartest Credit Card")
    st.write(
        "SafiCard uses **artificial intelligence** to protect every "
        "transaction in real time. Powered by XGBoost trained on "
        "**555,719 transactions** with fraud detection ROC-AUC of **0.9945**."
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔒 Secure**\nEvery transaction analysed instantly")
    with col2:
        st.markdown("**⚡ Fast**\nDecision in milliseconds")
    with col3:
        st.markdown("**🇰🇪 Local**\nBuilt for Kenyan merchants")

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("📝 Apply for Card", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()
    with col_b:
        if st.button("🔑 Login", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()
    with col_c:
        if st.button("🔧 Admin", use_container_width=True):
            st.session_state.page = 'admin'
            st.rerun()

    st.markdown("---")
    st.caption(
        "Final Year Project | BSc. Data Science | JKUAT | "
        "SafiCard Fraud Detection System | Model: XGBoost Tuned"
    )

# ============================================================
# PAGE: REGISTER
# ============================================================
def page_register():
    st.markdown('''
        <div class="safi-header">
            💳 Apply for SafiCard
            <div class="safi-tagline">Takes less than 2 minutes</div>
        </div>
    ''', unsafe_allow_html=True)

    # Show new card details if just registered
    if st.session_state.show_card and st.session_state.new_card_details:
        details = st.session_state.new_card_details
        st.success("🎉 SafiCard Approved! Save your card details below.")
        st.markdown(f"""
            <div class="credit-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="font-size:20px;font-weight:bold;">SafiCard</div>
                    <div style="font-size:16px;font-weight:bold;">
                        {"VISA" if details['card_type']=="Visa" else "●● MASTERCARD"}
                    </div>
                </div>
                <div style="font-size:22px;letter-spacing:4px;margin:20px 0 10px 0;">
                    {details['card_number']}
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <div>
                        <div style="font-size:10px;opacity:0.7;">CARD HOLDER</div>
                        <div style="font-size:14px;">{details['name'].upper()}</div>
                    </div>
                    <div>
                        <div style="font-size:10px;opacity:0.7;">EXPIRES</div>
                        <div style="font-size:14px;">{details['expiry']}</div>
                    </div>
                    <div>
                        <div style="font-size:10px;opacity:0.7;">CREDIT LIMIT</div>
                        <div style="font-size:14px;">
                            KES {details['credit_limit']:,.0f}
                        </div>
                    </div>
                </div>
                <div style="margin-top:15px;font-size:11px;opacity:0.7;">
                    ⚠️ Save your full card number — it will be masked after this screen
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("✅ Continue to Dashboard", use_container_width=True):
            st.session_state.show_card        = False
            st.session_state.new_card_details = None
            st.session_state.page             = 'dashboard'
            st.rerun()
        return

    # Registration form
    st.markdown("#### Personal Details")
    name       = st.text_input("Full Name")
    age        = st.number_input("Age", min_value=18, max_value=100, value=25)
    gender     = st.selectbox("Gender", ["M", "F"])
    occupation = st.selectbox("Occupation", list(OCCUPATIONS.keys()))
    county     = st.selectbox("County", list(COUNTIES.keys()))

    st.markdown("#### Card Settings")
    card_type    = st.selectbox("Card Type", ["Visa", "Mastercard"])
    credit_limit = st.number_input(
        "Credit Limit (KES)",
        min_value=5000.0, max_value=1000000.0,
        value=50000.0, step=1000.0
    )

    st.markdown("#### Security")
    pin         = st.text_input("Set Card PIN (4 digits)",
                                 type="password", max_chars=4)
    pin_confirm = st.text_input("Confirm PIN",
                                 type="password", max_chars=4)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Apply Now", use_container_width=True):
            if not name.strip():
                st.error("Please enter your full name.")
            elif name_exists(name.strip()):
                st.error("Name already registered. Please login.")
            elif len(pin) != 4 or not pin.isdigit():
                st.error("PIN must be exactly 4 digits.")
            elif pin != pin_confirm:
                st.error("PINs do not match.")
            else:
                user_id, card_number, expiry = create_user(
                    name.strip(), age, gender, occupation,
                    county, credit_limit, card_type, pin
                )
                # Store user in session
                st.session_state.user = {
                    'id':           user_id,
                    'name':         name.strip(),
                    'age':          age,
                    'gender':       gender,
                    'occupation':   occupation,
                    'county':       county,
                    'card_number':  str(card_number),
                    'card_type':    str(card_type),
                    'expiry':       str(expiry),
                    'credit_limit': credit_limit,
                    'balance':      credit_limit,
                }
                # Store card details to show
                st.session_state.new_card_details = {
                    'name':         name.strip(),
                    'card_number':  str(card_number),
                    'card_type':    str(card_type),
                    'expiry':       str(expiry),
                    'credit_limit': credit_limit,
                }
                st.session_state.logged_in = True
                st.session_state.show_card = True
                st.rerun()
    with col2:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 'landing'
            st.rerun()

# ============================================================
# PAGE: LOGIN
# ============================================================
def page_login():
    st.markdown('''
        <div class="safi-header">
            💳 SafiCard Login
            <div class="safi-tagline">Welcome back</div>
        </div>
    ''', unsafe_allow_html=True)

    name = st.text_input("Full Name")
    pin  = st.text_input("Card PIN", type="password", max_chars=4)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔑 Login", use_container_width=True):
            user = verify_pin(name.strip(), pin)
            if user:
                st.session_state.user = {
                    'id':           user['id'],
                    'name':         str(user['name']),
                    'age':          int(user['age']),
                    'gender':       str(user['gender']),
                    'occupation':   str(user['occupation']),
                    'county':       str(user['county']),
                    'card_number':  str(user['card_number']),
                    'card_type':    str(user['card_type']),
                    'expiry':       str(user['expiry']),
                    'credit_limit': float(user['credit_limit']),
                    'balance':      float(user['balance']),
                }
                st.session_state.logged_in = True
                st.session_state.page      = 'dashboard'
                st.rerun()
            else:
                st.error("Incorrect name or PIN.")
    with col2:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 'landing'
            st.rerun()

# ============================================================
# PAGE: DASHBOARD
# ============================================================
def page_dashboard():
    refresh_user()
    user = st.session_state.user

    st.markdown('''
        <div class="safi-header">
            💳 SafiCard Dashboard
            <div class="safi-tagline">Salama kila wakati</div>
        </div>
    ''', unsafe_allow_html=True)

    # Visual credit card
    card_logo = "VISA" if str(user.get('card_type','')) == 'Visa' else "●● MASTERCARD"
    st.markdown(f"""
        <div class="credit-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="font-size:20px;font-weight:bold;">SafiCard</div>
                <div style="font-size:16px;font-weight:bold;">{card_logo}</div>
            </div>
            <div style="font-size:22px;letter-spacing:4px;margin:20px 0 10px 0;">
                {mask_card(user.get('card_number','0000 0000 0000 0000'))}
            </div>
            <div style="display:flex;justify-content:space-between;">
                <div>
                    <div style="font-size:10px;opacity:0.7;">CARD HOLDER</div>
                    <div style="font-size:14px;">{str(user['name']).upper()}</div>
                </div>
                <div>
                    <div style="font-size:10px;opacity:0.7;">EXPIRES</div>
                    <div style="font-size:14px;">{user.get('expiry','--/--')}</div>
                </div>
                <div>
                    <div style="font-size:10px;opacity:0.7;">AVAILABLE CREDIT</div>
                    <div style="font-size:14px;font-weight:bold;">
                        KES {float(user['balance']):,.2f}
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Last result
    if st.session_state.last_result:
        r = st.session_state.last_result
        if r['fraud']:
            st.error(
                f"🚨 Last transaction BLOCKED — {r['merchant']} | "
                f"KES {float(r['amount']):,.2f} | "
                f"Risk: {float(r['prob'])*100:.1f}%"
            )
        else:
            st.success(
                f"✅ Last transaction APPROVED — {r['merchant']} | "
                f"KES {float(r['amount']):,.2f} | "
                f"Risk: {float(r['prob'])*100:.1f}%"
            )
        st.session_state.last_result = None

    if st.button("💳 Make a Payment", use_container_width=True):
        st.session_state.page = 'pay'
        st.rerun()

    # Stats
    stats = get_user_stats(user['id'])
    st.markdown("### 📊 Account Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", stats.get('total') or 0)
    c2.metric("Approved",           stats.get('approved') or 0)
    c3.metric("Blocked",            stats.get('blocked') or 0)
    c4.metric("Total Spent",
              f"KES {float(stats.get('total_spent') or 0):,.0f}")

    # Transaction history
    st.markdown("### 📋 Transaction History")
    txns = get_user_transactions(user['id'], limit=10)

    if not txns:
        st.info("No transactions yet. Make your first payment!")
    else:
        for txn in txns:
            css  = "txn-blocked" if txn['is_fraud'] else "txn-approved"
            icon = "🚨 BLOCKED"  if txn['is_fraud'] else "✅ APPROVED"
            cat  = str(txn['category']).replace('_',' ').title()
            st.markdown(f"""
                <div class="{css}">
                    <b>{icon}</b> — {txn['merchant']}<br>
                    <b>KES {float(txn['amount_kes']):,.2f}</b>
                    &nbsp;|&nbsp; {cat}
                    &nbsp;|&nbsp; Risk: {float(txn['fraud_prob'])*100:.1f}%
                    &nbsp;|&nbsp; {txn['timestamp']}
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user      = None
        st.session_state.page      = 'landing'
        st.rerun()

# ============================================================
# PAGE: PAY
# ============================================================
def page_pay():
    user = st.session_state.user

    st.markdown('''
        <div class="safi-header">
            💳 SafiCard Payment
            <div class="safi-tagline">Secure checkout</div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown(
        f"**{user['name']}** &nbsp;|&nbsp; "
        f"{mask_card(user.get('card_number',''))} &nbsp;|&nbsp; "
        f"Available: **KES {float(user['balance']):,.2f}**"
    )
    st.markdown("---")

    merchant = st.selectbox("Merchant", list(MERCHANTS.keys()))
    amount   = st.number_input(
        "Amount (KES)",
        min_value=1.0,
        max_value=float(user['balance']),
        value=500.0,
        step=1.0
    )

    cat_display = MERCHANTS[merchant][0].replace("_"," ").title()
    st.caption(f"Category: {cat_display}")

    # Demo mode
    with st.expander("🎯 Demo Mode (Presentation Use Only)"):
        demo_mode = st.checkbox("Enable Demo Mode")
        if demo_mode:
            sim_hour   = st.slider("Simulate Hour of Day", 0, 23, 23)
            multiplier = st.slider("Amount Multiplier", 1, 20, 1)
            st.caption(
                "⚠️ Simulates different time and amount conditions. "
                "For demonstration purposes only."
            )
        else:
            sim_hour   = None
            multiplier = 1

    pin = st.text_input("Enter Card PIN to confirm",
                         type="password", max_chars=4)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💳 Pay Now", use_container_width=True):
            if not pin:
                st.error("Please enter your PIN.")
            else:
                verified = verify_pin(user['name'], pin)
                if not verified:
                    st.error("❌ Wrong PIN. Transaction cancelled.")
                else:
                    with st.spinner("🔍 Checking transaction for fraud..."):
                        prob, is_fraud, category_raw = check_fraud(
                            user, merchant, amount,
                            sim_hour=sim_hour if demo_mode else None,
                            amount_multiplier=multiplier if demo_mode else 1
                        )

                    save_transaction(
                        user['id'], merchant, category_raw,
                        amount, prob, is_fraud,
                        "BLOCKED" if is_fraud else "APPROVED"
                    )

                    st.session_state.last_result = {
                        'merchant': merchant,
                        'amount':   amount,
                        'prob':     prob,
                        'fraud':    is_fraud,
                    }

                    if is_fraud:
                        st.error("🚨 Transaction BLOCKED")
                        st.warning(
                            f"SafiCard's fraud detection system flagged "
                            f"this transaction with a risk score of "
                            f"**{prob*100:.1f}%** "
                            f"(threshold: {float(threshold)*100:.0f}%). "
                            f"**No amount was deducted from your card.**"
                        )
                        st.markdown("**Why was this flagged?**")
                        use_hour = int(sim_hour) if (demo_mode and sim_hour is not None) \
                                   else datetime.now().hour
                        if (amount * multiplier) / 130 > 200:
                            st.write("🔴 Unusually high transaction amount")
                        if MERCHANTS[merchant][0] in ['shopping_net','misc_net','grocery_net']:
                            st.write("🔴 High-risk online merchant category")
                        if use_hour >= 22 or use_hour <= 3:
                            st.write("🔴 Transaction at high-risk hour (late night)")
                        st.write(
                            f"🔴 Combined risk score of {prob*100:.1f}% "
                            f"exceeds SafiCard threshold of "
                            f"{float(threshold)*100:.0f}%"
                        )
                    else:
                        new_balance = float(user['balance']) - float(amount)
                        update_balance(user['id'], new_balance)
                        st.session_state.user['balance'] = new_balance
                        st.success(
                            f"✅ Payment of **KES {amount:,.2f}** "
                            f"to **{merchant}** was successful!"
                        )
                        st.info(
                            f"Risk score: **{prob*100:.1f}%** — "
                            f"Below SafiCard threshold of "
                            f"{float(threshold)*100:.0f}%. Approved."
                        )
                        st.balloons()

                    # Risk gauge
                    risk_pct = int(prob * 100)
                    color    = ("#2E7D32" if prob < 0.3
                                else "#F57F17" if prob < 0.6
                                else "#C62828")
                    label    = ("LOW RISK" if prob < 0.3
                                else "MEDIUM RISK" if prob < 0.6
                                else "HIGH RISK")
                    st.markdown("**Fraud Risk Gauge**")
                    st.markdown(f"""
                        <div style='background:#e0e0e0;border-radius:10px;padding:4px;margin-top:10px;'>
                            <div style='background:{color};
                                        width:{max(risk_pct,8)}%;
                                        border-radius:10px;padding:10px;
                                        text-align:center;color:white;
                                        font-weight:bold;'>
                                {label} — {risk_pct}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

    with col2:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()

    st.markdown("---")
    st.caption(
        "🔒 Protected by SafiCard AI Fraud Detection | "
        "XGBoost Model | ROC-AUC: 0.9945 | "
        "Final Year Project — BSc. Data Science, JKUAT"
    )

# ============================================================
# PAGE: ADMIN
# ============================================================
def page_admin():
    st.markdown('''
        <div class="safi-header">
            🔧 SafiCard Admin Portal
            <div class="safi-tagline">System Overview</div>
        </div>
    ''', unsafe_allow_html=True)

    admin_pin = st.text_input("Admin PIN", type="password", max_chars=4)

    if admin_pin != "0000":
        st.warning("Enter admin PIN to continue (default: 0000)")
        if st.button("← Back"):
            st.session_state.page = 'landing'
            st.rerun()
        return

    st.success("✅ Admin access granted")

    # System stats
    stats = get_system_stats()
    st.markdown("### 📊 System Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Users",        stats.get('total_users') or 0)
    c2.metric("Total Transactions", stats.get('total_txns') or 0)
    c3.metric("Approved",           stats.get('approved') or 0)
    c4.metric("Blocked",            stats.get('blocked') or 0)
    c5.metric("Total Volume",
              f"KES {float(stats.get('total_volume') or 0):,.0f}")

    total_txns = stats.get('total_txns') or 0
    if total_txns:
        fraud_rate = (stats.get('blocked') or 0) / total_txns * 100
        st.markdown(f"**System Fraud Rate: {fraud_rate:.1f}%**")

    st.markdown("---")
    tab1, tab2 = st.tabs(["📋 All Transactions", "👥 All Cardholders"])

    with tab1:
        txns = get_all_transactions()
        if not txns:
            st.info("No transactions yet.")
        else:
            df = pd.DataFrame(txns)
            df['Status']     = df['is_fraud'].apply(
                lambda x: '🚨 BLOCKED' if x else '✅ APPROVED'
            )
            df['Risk Score'] = df['fraud_prob'].apply(
                lambda x: f"{float(x)*100:.1f}%"
            )
            df['Card'] = df['card_number'].apply(mask_card)
            st.dataframe(
                df[['timestamp','user_name','Card','merchant',
                    'category','amount_kes','Risk Score','Status'
                ]].rename(columns={
                    'timestamp':  'Time',
                    'user_name':  'Cardholder',
                    'merchant':   'Merchant',
                    'category':   'Category',
                    'amount_kes': 'Amount (KES)',
                }),
                use_container_width=True
            )

    with tab2:
        users = get_all_users()
        if not users:
            st.info("No cardholders yet.")
        else:
            df_u = pd.DataFrame(users)
            df_u['Card'] = df_u['card_number'].apply(mask_card)
            st.dataframe(
                df_u[['name','age','gender','occupation','county',
                       'Card','card_type','expiry',
                       'credit_limit','balance','created_at'
                ]].rename(columns={
                    'name':         'Name',
                    'age':          'Age',
                    'gender':       'Gender',
                    'occupation':   'Occupation',
                    'county':       'County',
                    'card_type':    'Type',
                    'expiry':       'Expiry',
                    'credit_limit': 'Credit Limit (KES)',
                    'balance':      'Available (KES)',
                    'created_at':   'Registered',
                }),
                use_container_width=True
            )

    st.markdown("---")
    if st.button("← Back to Landing", use_container_width=True):
        st.session_state.page = 'landing'
        st.rerun()

# ============================================================
# ROUTER
# ============================================================
if st.session_state.page == 'admin':
    page_admin()
elif not st.session_state.logged_in:
    if st.session_state.page == 'register':
        page_register()
    elif st.session_state.page == 'login':
        page_login()
    else:
        page_landing()
else:
    if st.session_state.page == 'pay':
        page_pay()
    else:
        page_dashboard()