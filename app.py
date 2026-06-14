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
    page_title="SafiCard",
    page_icon="💳",
    layout="centered"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
    <style>
    /* General */
   .block-container { 
    padding-top: 0rem !important; 
    padding-bottom: 1rem; 
    max-width: 800px;
}
[data-testid="stAppViewContainer"] { padding-top: 0rem !important; }
[data-testid="stHeader"] { display: none !important; }

    /* Header */
    .safi-header {
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        padding: 18px 25px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 18px;
        letter-spacing: 1px;
    }
    .safi-tagline {
        font-size: 12px;
        opacity: 0.85;
        margin-top: 4px;
        font-weight: normal;
    }

    /* Credit Card */
    .credit-card {
        background: linear-gradient(135deg, #1B5E20, #388E3C);
        padding: 22px 25px;
        border-radius: 16px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        font-family: 'Courier New', monospace;
    }

    /* Transaction cards */
    .txn-approved {
        background-color: #E8F5E9;
        padding: 12px 15px;
        border-radius: 8px;
        border-left: 5px solid #2E7D32;
        margin: 6px 0;
        color: #000;
        font-size: 14px;
    }
    .txn-blocked {
        background-color: #FFEBEE;
        padding: 12px 15px;
        border-radius: 8px;
        border-left: 5px solid #C62828;
        margin: 6px 0;
        color: #000;
        font-size: 14px;
    }

    /* Result cards */
    .result-approved {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #2E7D32;
        margin: 15px 0;
    }
    .result-blocked {
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #C62828;
        margin: 15px 0;
    }

    /* Risk gauge */
    .gauge-container {
        background: #e0e0e0;
        border-radius: 10px;
        padding: 4px;
        margin: 10px 0;
    }

    /* Stat box */
    .stat-box {
        background: #F1F8E9;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #AED581;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1B5E20;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2E7D32;
        color: white;
    }

    /* Input fields */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #A5D6A7;
    }
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #A5D6A7;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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

MERCHANT_ICONS = {
    "Naivas Supermarket":        "🛒",
    "Quickmart":                 "🛒",
    "Carrefour Kenya":           "🏪",
    "Jumia Kenya":               "📦",
    "Netflix":                   "🎬",
    "Showmax":                   "📺",
    "Java House":                "☕",
    "KFC Kenya":                 "🍗",
    "Glovo Kenya":               "🛵",
    "Uber Kenya":                "🚗",
    "KPLC Prepaid":              "💡",
    "Total Energies":            "⛽",
    "Goodlife Pharmacy":         "💊",
    "SGR Madaraka Express":      "🚂",
    "Nairobi Cinema":            "🎭",
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
    'pay_result':       None,
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

def render_header(title, tagline="Salama kila wakati"):
    st.markdown(f'''
        <div class="safi-header">
            💳 {title}
            <div class="safi-tagline">{tagline}</div>
        </div>
    ''', unsafe_allow_html=True)

def render_card(user):
    card_logo = "VISA" if str(user.get('card_type','')) == 'Visa' else "●● MASTERCARD"
    st.markdown(f"""
        <div class="credit-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="font-size:18px;font-weight:bold;letter-spacing:2px;">SafiCard</div>
                <div style="font-size:15px;font-weight:bold;">{card_logo}</div>
            </div>
            <div style="margin:8px 0;">
                <div style="width:40px;height:30px;background:#F9A825;
                            border-radius:4px;opacity:0.9;"></div>
            </div>
            <div style="font-size:20px;letter-spacing:4px;margin:12px 0 8px 0;">
                {mask_card(user.get('card_number','0000 0000 0000 0000'))}
            </div>
            <div style="display:flex;justify-content:space-between;align-items:flex-end;">
                <div>
                    <div style="font-size:9px;opacity:0.7;text-transform:uppercase;
                                letter-spacing:1px;">Card Holder</div>
                    <div style="font-size:13px;font-weight:bold;">
                        {str(user['name']).upper()}
                    </div>
                </div>
                <div>
                    <div style="font-size:9px;opacity:0.7;text-transform:uppercase;
                                letter-spacing:1px;">Expires</div>
                    <div style="font-size:13px;">{user.get('expiry','--/--')}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:9px;opacity:0.7;text-transform:uppercase;
                                letter-spacing:1px;">Available Credit</div>
                    <div style="font-size:16px;font-weight:bold;">
                        KES {float(user['balance']):,.2f}
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_gauge(prob):
    risk_pct = int(prob * 100)
    color    = ("#2E7D32" if prob < 0.3
                else "#F57F17" if prob < 0.6
                else "#C62828")
    label    = ("LOW RISK" if prob < 0.3
                else "MEDIUM RISK" if prob < 0.6
                else "HIGH RISK")
    st.markdown(f"""
        <div class="gauge-container">
            <div style='background:{color};
                        width:{max(risk_pct, 8)}%;
                        border-radius:8px;
                        padding:8px;
                        text-align:center;
                        color:white;
                        font-weight:bold;
                        font-size:13px;'>
                {label} — {risk_pct}%
            </div>
        </div>
    """, unsafe_allow_html=True)

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
    render_header("SafiCard", "Salama kila wakati — Safe every time")

    st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px 0;'>
            <div style='font-size:18px; color:#1B5E20; font-weight:bold;'>
                Kenya's Smartest Credit Card
            </div>
            <div style='font-size:14px; color:#555; margin-top:8px;'>
                Every transaction protected by AI fraud detection
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style='text-align:center; padding:15px;
                        background:#E8F5E9; border-radius:10px;'>
                <div style='font-size:28px;'>🔒</div>
                <div style='font-weight:bold; color:#1B5E20;'>Secure</div>
                <div style='font-size:12px; color:#555;'>
                    AI analyses every transaction instantly
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style='text-align:center; padding:15px;
                        background:#E8F5E9; border-radius:10px;'>
                <div style='font-size:28px;'>⚡</div>
                <div style='font-weight:bold; color:#1B5E20;'>Fast</div>
                <div style='font-size:12px; color:#555;'>
                    Decision in milliseconds
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style='text-align:center; padding:15px;
                        background:#E8F5E9; border-radius:10px;'>
                <div style='font-size:28px;'>🇰🇪</div>
                <div style='font-weight:bold; color:#1B5E20;'>Local</div>
                <div style='font-size:12px; color:#555;'>
                    Built for Kenyan merchants
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📝 Apply for SafiCard", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()
    with col_b:
        if st.button("🔑 Login to My Card", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats bar
    st.markdown("""
        <div style='background:#1B5E20; border-radius:10px; padding:12px;
                    display:flex; justify-content:space-around; text-align:center;'>
            <div style='color:white;'>
                <div style='font-size:18px; font-weight:bold;'>555,719</div>
                <div style='font-size:11px; opacity:0.8;'>Transactions Analysed</div>
            </div>
            <div style='color:white;'>
                <div style='font-size:18px; font-weight:bold;'>99.45%</div>
                <div style='font-size:11px; opacity:0.8;'>Detection Accuracy</div>
            </div>
            <div style='color:white;'>
                <div style='font-size:18px; font-weight:bold;'>88%</div>
                <div style='font-size:11px; opacity:0.8;'>Fraud Caught</div>
            </div>
            <div style='color:white;'>
                <div style='font-size:18px; font-weight:bold;'>0.19%</div>
                <div style='font-size:11px; opacity:0.8;'>False Block Rate</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align:center;'>
            <a href='?page=admin' style='color:#888; font-size:12px;
               text-decoration:none;'>Admin Portal</a>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🔧 Admin Portal", use_container_width=False):
        st.session_state.page = 'admin'
        st.rerun()

    st.markdown("---")
    st.caption(
        "Final Year Project | BSc. Data Science | JKUAT | "
        "SafiCard — Powered by XGBoost (ROC-AUC: 0.9945)"
    )

# ============================================================
# PAGE: REGISTER
# ============================================================
def page_register():
    render_header("Apply for SafiCard", "Takes less than 2 minutes")

    # Show new card after registration
    if st.session_state.show_card and st.session_state.new_card_details:
        details = st.session_state.new_card_details
        st.success("🎉 Congratulations! Your SafiCard has been approved.")
        st.markdown("**Please save your card details below:**")

        card_logo = "VISA" if details['card_type'] == "Visa" else "●● MASTERCARD"
        st.markdown(f"""
            <div class="credit-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="font-size:18px;font-weight:bold;">SafiCard</div>
                    <div style="font-size:15px;font-weight:bold;">{card_logo}</div>
                </div>
                <div style="margin:8px 0;">
                    <div style="width:40px;height:30px;background:#F9A825;
                                border-radius:4px;opacity:0.9;"></div>
                </div>
                <div style="font-size:20px;letter-spacing:4px;margin:12px 0 8px 0;">
                    {details['card_number']}
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <div>
                        <div style="font-size:9px;opacity:0.7;">CARD HOLDER</div>
                        <div style="font-size:13px;font-weight:bold;">
                            {details['name'].upper()}
                        </div>
                    </div>
                    <div>
                        <div style="font-size:9px;opacity:0.7;">EXPIRES</div>
                        <div style="font-size:13px;">{details['expiry']}</div>
                    </div>
                    <div>
                        <div style="font-size:9px;opacity:0.7;">CREDIT LIMIT</div>
                        <div style="font-size:13px;">
                            KES {details['credit_limit']:,.0f}
                        </div>
                    </div>
                </div>
                <div style="margin-top:12px;font-size:11px;opacity:0.7;
                            border-top:1px solid rgba(255,255,255,0.3);
                            padding-top:8px;">
                    ⚠️ Save your full card number — it will be masked after this screen
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.info(
            f"Your card number is: **{details['card_number']}**\n\n"
            f"Card Type: {details['card_type']} | "
            f"Expires: {details['expiry']}"
        )

        if st.button("✅ Continue to My Dashboard", use_container_width=True):
            st.session_state.show_card        = False
            st.session_state.new_card_details = None
            st.session_state.page             = 'dashboard'
            st.rerun()
        return

    # Registration form
    st.markdown("#### 👤 Personal Details")
    st.caption("These details are stored securely and used for fraud detection.")

    col1, col2 = st.columns(2)
    with col1:
        name   = st.text_input("Full Name", placeholder="e.g. John Kamau")
        age    = st.number_input("Age", min_value=18, max_value=100, value=25)
        gender = st.selectbox("Gender", ["M", "F"])
    with col2:
        occupation = st.selectbox("Occupation", list(OCCUPATIONS.keys()))
        county     = st.selectbox("County", list(COUNTIES.keys()))

    st.markdown("#### 💳 Card Settings")
    col3, col4 = st.columns(2)
    with col3:
        card_type = st.selectbox("Card Type", ["Visa", "Mastercard"])
    with col4:
        credit_limit = st.number_input(
            "Credit Limit (KES)",
            min_value=5000.0, max_value=1000000.0,
            value=50000.0, step=1000.0
        )

    st.markdown("#### 🔐 Security")
    col5, col6 = st.columns(2)
    with col5:
        pin = st.text_input("Set Card PIN (4 digits)",
                             type="password", max_chars=4,
                             placeholder="****")
    with col6:
        pin_confirm = st.text_input("Confirm PIN",
                                     type="password", max_chars=4,
                                     placeholder="****")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Apply Now", use_container_width=True):
            if not name.strip():
                st.error("Please enter your full name.")
            elif name_exists(name.strip()):
                st.error("Name already registered. Please login instead.")
            elif len(pin) != 4 or not pin.isdigit():
                st.error("PIN must be exactly 4 digits.")
            elif pin != pin_confirm:
                st.error("PINs do not match. Please try again.")
            else:
                user_id, card_number, expiry = create_user(
                    name.strip(), age, gender, occupation,
                    county, credit_limit, card_type, pin
                )
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
    with col_b:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 'landing'
            st.rerun()

# ============================================================
# PAGE: LOGIN
# ============================================================
def page_login():
    render_header("SafiCard Login", "Welcome back")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("#### Enter your details")
        name = st.text_input("Full Name", placeholder="As registered")
        pin  = st.text_input("Card PIN", type="password",
                              max_chars=4, placeholder="****")
        st.markdown("<br>", unsafe_allow_html=True)

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
                st.error("❌ Incorrect name or PIN. Please try again.")

        if st.button("← Back", use_container_width=True):
            st.session_state.page = 'landing'
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center; font-size:13px; color:#666;'>"
            "Don't have a card? "
            "</div>",
            unsafe_allow_html=True
        )
        if st.button("Apply for SafiCard →", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()

# ============================================================
# PAGE: DASHBOARD
# ============================================================
def page_dashboard():
    refresh_user()
    user = st.session_state.user

    render_header("SafiCard", f"Welcome back, {user['name'].split()[0]}")

    # Credit card visual
    render_card(user)

    # Last transaction result
    if st.session_state.last_result:
        r = st.session_state.last_result
        icon = MERCHANT_ICONS.get(r['merchant'], '💳')
        if r['fraud']:
            st.markdown(f"""
                <div class="txn-blocked">
                    🚨 <b>Blocked</b> — {icon} {r['merchant']} |
                    KES {float(r['amount']):,.2f} |
                    Risk: {float(r['prob'])*100:.1f}%
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="txn-approved">
                    ✅ <b>Approved</b> — {icon} {r['merchant']} |
                    KES {float(r['amount']):,.2f} |
                    Risk: {float(r['prob'])*100:.1f}%
                </div>
            """, unsafe_allow_html=True)
        st.session_state.last_result = None

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💳 Make a Payment", use_container_width=True):
            st.session_state.page = 'pay'
            st.rerun()
    with col2:
        if st.button("📋 Transaction History", use_container_width=True):
            st.session_state.page = 'history'
            st.rerun()

    # Account summary
    stats = get_user_stats(user['id'])
    st.markdown("### 📊 Account Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="stat-box">
                <div style='font-size:22px;font-weight:bold;color:#1B5E20;'>
                    {stats.get('total') or 0}
                </div>
                <div style='font-size:12px;color:#555;'>Total Transactions</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="stat-box">
                <div style='font-size:22px;font-weight:bold;color:#2E7D32;'>
                    {stats.get('approved') or 0}
                </div>
                <div style='font-size:12px;color:#555;'>Approved</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="stat-box">
                <div style='font-size:22px;font-weight:bold;color:#C62828;'>
                    {stats.get('blocked') or 0}
                </div>
                <div style='font-size:12px;color:#555;'>Blocked</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="stat-box">
                <div style='font-size:16px;font-weight:bold;color:#1B5E20;'>
                    KES {float(stats.get('total_spent') or 0):,.0f}
                </div>
                <div style='font-size:12px;color:#555;'>Total Spent</div>
            </div>
        """, unsafe_allow_html=True)

    # Recent transactions preview
    st.markdown("### 🕐 Recent Activity")
    txns = get_user_transactions(user['id'], limit=3)

    if not txns:
        st.info("No transactions yet. Make your first payment!")
    else:
        for txn in txns:
            icon = MERCHANT_ICONS.get(txn['merchant'], '💳')
            css  = "txn-blocked" if txn['is_fraud'] else "txn-approved"
            badge = "🚨 Blocked" if txn['is_fraud'] else "✅ Approved"
            st.markdown(f"""
                <div class="{css}">
                    {icon} <b>{txn['merchant']}</b>
                    &nbsp;—&nbsp; KES {float(txn['amount_kes']):,.2f}
                    &nbsp;|&nbsp; {badge}
                    &nbsp;|&nbsp; Risk: {float(txn['fraud_prob'])*100:.1f}%
                    <br>
                    <span style='font-size:11px;color:#666;'>
                        {txn['timestamp']}
                    </span>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user      = None
        st.session_state.page      = 'landing'
        st.rerun()

# ============================================================
# PAGE: TRANSACTION HISTORY
# ============================================================
def page_history():
    refresh_user()
    user = st.session_state.user

    render_header("Transaction History", f"{user['name']}")

    txns = get_user_transactions(user['id'], limit=50)

    if not txns:
        st.info("No transactions yet.")
    else:
        # Summary
        total   = len(txns)
        blocked = sum(1 for t in txns if t['is_fraud'])
        st.markdown(
            f"Showing **{total}** transactions — "
            f"**{total-blocked}** approved, **{blocked}** blocked"
        )
        st.markdown("---")

        for txn in txns:
            icon  = MERCHANT_ICONS.get(txn['merchant'], '💳')
            css   = "txn-blocked" if txn['is_fraud'] else "txn-approved"
            badge = "🚨 Blocked" if txn['is_fraud'] else "✅ Approved"
            cat   = str(txn['category']).replace('_', ' ').title()
            st.markdown(f"""
                <div class="{css}">
                    <div style='display:flex;justify-content:space-between;
                                align-items:center;'>
                        <div>
                            {icon} <b>{txn['merchant']}</b>
                            &nbsp;—&nbsp;
                            <span style='font-size:13px;color:#444;'>{cat}</span>
                        </div>
                        <div style='font-weight:bold;font-size:15px;'>
                            KES {float(txn['amount_kes']):,.2f}
                        </div>
                    </div>
                    <div style='margin-top:5px;font-size:12px;color:#555;
                                display:flex;justify-content:space-between;'>
                        <span>{badge} &nbsp;|&nbsp; Risk: {float(txn['fraud_prob'])*100:.1f}%</span>
                        <span>{txn['timestamp']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("← Back to Dashboard", use_container_width=True):
        st.session_state.page = 'dashboard'
        st.rerun()

# ============================================================
# PAGE: PAY
# ============================================================
def page_pay():
    refresh_user()
    user = st.session_state.user

    render_header("Make a Payment", "Secure checkout")

    # Show payment result if just processed
    if st.session_state.pay_result:
        r    = st.session_state.pay_result
        icon = MERCHANT_ICONS.get(r['merchant'], '💳')

        if not r['fraud']:
            st.markdown(f"""
                <div class="result-approved">
                    <div style='font-size:48px;'>✅</div>
                    <div style='font-size:22px;font-weight:bold;
                                color:#1B5E20;margin:10px 0;'>
                        Payment Successful
                    </div>
                    <div style='font-size:16px;color:#333;'>
                        {icon} {r['merchant']}
                    </div>
                    <div style='font-size:28px;font-weight:bold;
                                color:#1B5E20;margin:10px 0;'>
                        KES {float(r['amount']):,.2f}
                    </div>
                    <div style='font-size:13px;color:#555;'>
                        Fraud Risk Score: {float(r['prob'])*100:.1f}% — Safe
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
                <div class="result-blocked">
                    <div style='font-size:48px;'>🚨</div>
                    <div style='font-size:22px;font-weight:bold;
                                color:#C62828;margin:10px 0;'>
                        Transaction Blocked
                    </div>
                    <div style='font-size:16px;color:#333;'>
                        {icon} {r['merchant']}
                    </div>
                    <div style='font-size:28px;font-weight:bold;
                                color:#C62828;margin:10px 0;'>
                        KES {float(r['amount']):,.2f}
                    </div>
                    <div style='font-size:13px;color:#555;'>
                        Fraud Risk Score: {float(r['prob'])*100:.1f}% —
                        Exceeds threshold of {float(threshold)*100:.0f}%
                    </div>
                    <div style='font-size:13px;color:#C62828;margin-top:8px;'>
                        Your money is safe. No amount was deducted.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Risk gauge
        st.markdown("**Fraud Risk Gauge**")
        render_gauge(r['prob'])

        # Reasons if blocked
        if r['fraud']:
            st.markdown("**Why was this flagged?**")
            reasons = r.get('reasons', [])
            for reason in reasons:
                st.write(reason)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💳 Make Another Payment", use_container_width=True):
                st.session_state.pay_result = None
                st.rerun()
        with col2:
            if st.button("🏠 Back to Dashboard", use_container_width=True):
                st.session_state.last_result = r
                st.session_state.pay_result  = None
                st.session_state.page        = 'dashboard'
                st.rerun()
        return

    # Payment form
    st.markdown(
        f"**Available Credit:** "
        f"<span style='color:#1B5E20;font-size:18px;font-weight:bold;'>"
        f"KES {float(user['balance']):,.2f}</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"Card: {mask_card(user.get('card_number',''))}")
    st.markdown("---")

    # Merchant selection with icons
    merchant_options = [
        f"{MERCHANT_ICONS.get(m, '💳')} {m}"
        for m in MERCHANTS.keys()
    ]
    selected = st.selectbox("Select Merchant", merchant_options)
    merchant = selected.split(" ", 1)[1]  # Remove icon prefix

    amount = st.number_input(
        "Amount (KES)",
        min_value=1.0,
        max_value=float(user['balance']),
        value=500.0,
        step=1.0,
        help="Enter the amount you want to pay"
    )

    cat_display = MERCHANTS[merchant][0].replace("_", " ").title()
    st.caption(f"Category: {cat_display}")

    # Demo mode — hidden in expander for presentation
    with st.expander("🎯 Demo Mode (Presentation Use Only)"):
        demo_mode = st.checkbox("Enable Demo Mode")
        if demo_mode:
            sim_hour   = st.slider("Simulate Hour of Day", 0, 23, 23)
            multiplier = st.slider("Amount Multiplier", 1, 20, 1)
            st.caption("⚠️ For demonstration purposes only.")
        else:
            sim_hour   = None
            multiplier = 1

    st.markdown("---")
    pin = st.text_input(
        "Enter Card PIN",
        type="password",
        max_chars=4,
        placeholder="Enter your 4-digit PIN to confirm payment"
    )

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
                    with st.spinner("🔍 Processing payment..."):
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

                    if not is_fraud:
                        new_balance = float(user['balance']) - float(amount)
                        update_balance(user['id'], new_balance)
                        st.session_state.user['balance'] = new_balance

                    # Build reasons
                    reasons = []
                    use_hour = int(sim_hour) if (demo_mode and sim_hour is not None) \
                               else datetime.now().hour
                    if (amount * (multiplier if demo_mode else 1)) / 130 > 200:
                        reasons.append("🔴 Unusually high transaction amount")
                    if MERCHANTS[merchant][0] in ['shopping_net','misc_net','grocery_net']:
                        reasons.append("🔴 High-risk online merchant category")
                    if use_hour >= 22 or use_hour <= 3:
                        reasons.append("🔴 Transaction at high-risk hour (late night)")
                    reasons.append(
                        f"🔴 Combined risk score of {prob*100:.1f}% "
                        f"exceeds threshold of {float(threshold)*100:.0f}%"
                    )

                    st.session_state.pay_result = {
                        'merchant': merchant,
                        'amount':   amount,
                        'prob':     prob,
                        'fraud':    is_fraud,
                        'reasons':  reasons if is_fraud else [],
                    }
                    st.rerun()

    with col2:
        if st.button("← Cancel", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()

    st.markdown("---")
    st.caption(
        "🔒 Protected by SafiCard AI | XGBoost Model | ROC-AUC: 0.9945 | "
        "Final Year Project — BSc. Data Science, JKUAT"
    )

# ============================================================
# PAGE: ADMIN
# ============================================================
def page_admin():
    render_header("SafiCard Admin Portal", "System Overview")

    admin_pin = st.text_input("Admin PIN", type="password", max_chars=4)

    if admin_pin != "0000":
        st.warning("Enter admin PIN to continue (default: 0000)")
        if st.button("← Back to Home"):
            st.session_state.page = 'landing'
            st.rerun()
        return

    st.success("✅ Admin access granted")

    # System stats
    stats = get_system_stats()
    st.markdown("### 📊 System Overview")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Users",        stats.get('total_users') or 0)
    col2.metric("Total Transactions", stats.get('total_txns') or 0)
    col3.metric("Approved",           stats.get('approved') or 0)
    col4.metric("Blocked",            stats.get('blocked') or 0)
    col5.metric("Volume",
                f"KES {float(stats.get('total_volume') or 0):,.0f}")

    total_txns = stats.get('total_txns') or 0
    if total_txns:
        fraud_rate = (stats.get('blocked') or 0) / total_txns * 100
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"**System Fraud Rate: {fraud_rate:.1f}%**"
            )
        with col2:
            st.markdown(
                f"**Model Threshold: {float(threshold)*100:.0f}%**"
            )

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
    if st.button("← Back to Home", use_container_width=True):
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
    elif st.session_state.page == 'history':
        page_history()
    else:
        page_dashboard()