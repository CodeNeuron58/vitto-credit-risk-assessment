import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config for a premium, wide dashboard layout
st.set_page_config(
    page_title="Vitto Credit Risk Portal",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Injection for Harmonious Theme & Glassmorphism
st.markdown("""
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    /* Expander card styling */
    .st-emotion-cache-1h95u2q {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
    }
    
    /* Clean Premium Card */
    .risk-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }
    
    /* Styled metric values */
    .prob-val {
        font-size: 4.5rem;
        font-weight: 700;
        line-height: 1;
        margin: 1rem 0;
    }
    
    /* Risk Badges */
    .badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-med {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Business Insight section */
    .driver-title {
        color: #a855f7;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .driver-item {
        background-color: #1f2937;
        border-left: 4px solid #6366f1;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        width: 100%;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model pickles safely
@st.cache_resource
def load_model_artifacts():
    app_dir = "app"
    model_path = os.path.join(app_dir, "xgb_model.pkl")
    scaler_path = os.path.join(app_dir, "scaler.pkl")
    features_path = os.path.join(app_dir, "feature_names.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        return None, None, None
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(features_path, "rb") as f:
        features = pickle.load(f)
        
    return model, scaler, features

xgb_model, scaler, feature_names = load_model_artifacts()

# Map UI text to numerical pay status codes
PAY_STATUS_MAP = {
    "Paid Duly / Early (-1)": -1,
    "No Consumption (-2)": -2,
    "Revolving / Minimum Pay (0)": 0,
    "1 Month Late (1)": 1,
    "2 Months Late (2)": 2,
    "3 Months Late (3)": 3,
    "4 Months Late (4)": 4,
    "5 Months Late (5)": 5,
    "6 Months Late (6)": 6,
    "7 Months Late (7)": 7,
    "8 Months Late (8)": 8,
}

# Mapping target details
EDUCATION_MAP = {"Graduate School": 1, "University": 2, "High School": 3, "Other": 4}
MARRIAGE_MAP = {"Married": 1, "Single": 2, "Other": 3}
SEX_MAP = {"Male": 1, "Female": 2}

# Main Application Layout
st.markdown("<div class='main-title'>Vitto Credit Risk Decisioning Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Real-Time Probability of Default Assessment & Risk Classifier</div>", unsafe_allow_html=True)

if xgb_model is None:
    st.error("⚠️ Model artifacts not found. Please run `python app/train_and_save.py` first to train and serialize the model.")
else:
    # Set up layout: Left side inputs, Right side decisions
    col_input, col_decision = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        st.subheader("📋 Borrower Information Panel")
        
        # Section 1: Demographics & Limit (Expandable)
        with st.expander("⚡ Demographic & Core Credit Details", expanded=True):
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                limit_bal = st.number_input("Credit Limit (NTD)", min_value=5000, max_value=1000000, value=100000, step=10000)
                age = st.slider("Borrower Age", min_value=18, max_value=80, value=35)
                sex = st.selectbox("Borrower Gender", list(SEX_MAP.keys()))
            with sub_col2:
                education = st.selectbox("Education Level", list(EDUCATION_MAP.keys()))
                marriage = st.selectbox("Marital Status", list(MARRIAGE_MAP.keys()))
        
        # Section 2: Repayment Delay Status (Expandable)
        with st.expander("📅 Repayment Status History (Last 6 Months)", expanded=False):
            st.markdown("<small>Select payment delays. (Note: PAY_0 is the most recent month, PAY_6 is 6 months ago)</small>", unsafe_allow_html=True)
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                pay_0_str = st.selectbox("Repayment Status in Sep (PAY_0)", list(PAY_STATUS_MAP.keys()), index=2)
                pay_2_str = st.selectbox("Repayment Status in Aug (PAY_2)", list(PAY_STATUS_MAP.keys()), index=2)
                pay_3_str = st.selectbox("Repayment Status in Jul (PAY_3)", list(PAY_STATUS_MAP.keys()), index=2)
            with p_col2:
                pay_4_str = st.selectbox("Repayment Status in Jun (PAY_4)", list(PAY_STATUS_MAP.keys()), index=2)
                pay_5_str = st.selectbox("Repayment Status in May (PAY_5)", list(PAY_STATUS_MAP.keys()), index=2)
                pay_6_str = st.selectbox("Repayment Status in Apr (PAY_6)", list(PAY_STATUS_MAP.keys()), index=2)
                
            pay_0 = PAY_STATUS_MAP[pay_0_str]
            pay_2 = PAY_STATUS_MAP[pay_2_str]
            pay_3 = PAY_STATUS_MAP[pay_3_str]
            pay_4 = PAY_STATUS_MAP[pay_4_str]
            pay_5 = PAY_STATUS_MAP[pay_5_str]
            pay_6 = PAY_STATUS_MAP[pay_6_str]
            
        # Section 3: Billing & Payments (Expandable)
        with st.expander("💵 Monthly Statements & Payments History", expanded=False):
            st.markdown("<small>Provide statements and previous payment amounts. Utilisation and pay ratios will be computed automatically.</small>", unsafe_allow_html=True)
            
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown("**Bill Statement Amount (NTD)**", unsafe_allow_html=True)
                bill_1 = st.number_input("September Bill (BILL_AMT1)", value=30000, step=5000)
                bill_2 = st.number_input("August Bill (BILL_AMT2)", value=28000, step=5000)
                bill_3 = st.number_input("July Bill (BILL_AMT3)", value=25000, step=5000)
                bill_4 = st.number_input("June Bill (BILL_AMT4)", value=22000, step=5000)
                bill_5 = st.number_input("May Bill (BILL_AMT5)", value=20000, step=5000)
                bill_6 = st.number_input("April Bill (BILL_AMT6)", value=18000, step=5000)
            with b_col2:
                st.markdown("**Previous Payment Amount (NTD)**", unsafe_allow_html=True)
                pay_1 = st.number_input("September Payment (PAY_AMT1)", value=3000, step=1000)
                pay_2 = st.number_input("August Payment (PAY_AMT2)", value=2500, step=1000)
                pay_3 = st.number_input("July Payment (PAY_AMT3)", value=2000, step=1000)
                pay_4 = st.number_input("June Payment (PAY_AMT4)", value=2000, step=1000)
                pay_5 = st.number_input("May Payment (PAY_AMT5)", value=1500, step=1000)
                pay_6 = st.number_input("April Payment (PAY_AMT6)", value=1500, step=1000)

        # Trigger Calculation
        assess_button = st.button("ASSESS CREDIT RISK")

    with col_decision:
        st.subheader("🎯 Risk Decision Center")
        
        # Calculate engineered metrics on the fly
        bills = [bill_1, bill_2, bill_3, bill_4, bill_5, bill_6]
        pays = [pay_1, pay_2, pay_3, pay_4, pay_5, pay_6]
        pay_statuses = [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]
        
        # 1. Utilisation Rate
        utils = [max(0, b) / limit_bal for b in bills]
        avg_util_rate = np.mean(utils)
        
        # 2. Payment Ratio
        ratios = []
        for b, p in zip(bills, pays):
            if b > 0:
                ratios.append(p / b)
        avg_pay_ratio = np.mean(ratios) if len(ratios) > 0 else 0.0
        
        # 3. Delinquency Months
        total_delay_months = sum(1 for status in pay_statuses if status > 0)
        
        if assess_button:
            # Map Categoricals to One-Hot Dummy columns
            sex_code = SEX_MAP[sex]
            edu_code = EDUCATION_MAP[education]
            mar_code = MARRIAGE_MAP[marriage]
            
            # Setup base dictionary with all features initialized to 0
            feature_dict = {feat: 0.0 for feat in feature_names}
            
            # Fill numeric continuous variables
            feature_dict['LIMIT_BAL'] = float(limit_bal)
            feature_dict['AGE'] = float(age)
            feature_dict['PAY_0'] = float(pay_0)
            feature_dict['PAY_2'] = float(pay_2)
            feature_dict['PAY_3'] = float(pay_3)
            feature_dict['PAY_4'] = float(pay_4)
            feature_dict['PAY_5'] = float(pay_5)
            feature_dict['PAY_6'] = float(pay_6)
            
            feature_dict['BILL_AMT1'] = float(bill_1)
            feature_dict['BILL_AMT2'] = float(bill_2)
            feature_dict['BILL_AMT3'] = float(bill_3)
            feature_dict['BILL_AMT4'] = float(bill_4)
            feature_dict['BILL_AMT5'] = float(bill_5)
            feature_dict['BILL_AMT6'] = float(bill_6)
            
            feature_dict['PAY_AMT1'] = float(pay_1)
            feature_dict['PAY_AMT2'] = float(pay_2)
            feature_dict['PAY_AMT3'] = float(pay_3)
            feature_dict['PAY_AMT4'] = float(pay_4)
            feature_dict['PAY_AMT5'] = float(pay_5)
            feature_dict['PAY_AMT6'] = float(pay_6)
            
            feature_dict['AVG_UTIL_RATE'] = float(avg_util_rate)
            feature_dict['AVG_PAY_RATIO'] = float(avg_pay_ratio)
            feature_dict['TOTAL_DELAY_MONTHS'] = float(total_delay_months)
            
            # One-hot encoded dummy triggers (using drop_first=True)
            if sex_code == 2:
                feature_dict['SEX_2'] = 1
            if edu_code == 2:
                feature_dict['EDUCATION_2'] = 1
            elif edu_code == 3:
                feature_dict['EDUCATION_3'] = 1
            elif edu_code == 4:
                feature_dict['EDUCATION_4'] = 1
                
            if mar_code == 2:
                feature_dict['MARRIAGE_2'] = 1
            elif mar_code == 3:
                feature_dict['MARRIAGE_3'] = 1
                
            # Convert dictionary to DataFrame with the exact feature order
            input_df = pd.DataFrame([feature_dict])[feature_names]
            
            # Scale the inputs
            input_scaled = scaler.transform(input_df)
            
            # Run Predict Proba
            prob = xgb_model.predict_proba(input_scaled)[0, 1]
            
            # Determine Risk Tier and formatting
            if prob < 0.25:
                risk_tier = "Low Risk"
                badge_class = "badge-low"
                tier_color = "#10b981"
            elif prob < 0.50:
                risk_tier = "Medium Risk"
                badge_class = "badge-med"
                tier_color = "#f59e0b"
            else:
                risk_tier = "High Risk"
                badge_class = "badge-high"
                tier_color = "#ef4444"
                
            # Display risk results card
            st.markdown(f"""
            <div class='risk-card'>
                <span class='badge {badge_class}'>{risk_tier}</span>
                <div class='prob-val' style='color: {tier_color}'>{prob*100:.1f}%</div>
                <div style='color: #9ca3af; font-size: 0.95rem; margin-top:-0.5rem;'>Calculated Probability of Default</div>
                
                <hr style='border-color: #1f2937; margin: 1.5rem 0;'>
                
                <div class='driver-title'>📊 Key Risk Drivers Identified:</div>
                <div class='driver-item'>
                    <strong>Delinquency History:</strong> Borrower has been late <strong>{total_delay_months}</strong> months in the past 6 months.
                </div>
                <div class='driver-item'>
                    <strong>Recent Repayment Status (PAY_0):</strong> September payment is classified as <strong>{pay_0_str}</strong>.
                </div>
                <div class='driver-item'>
                    <strong>Credit Utilisation Rate:</strong> Average utilization over the last 6 months is <strong>{avg_util_rate*100:.1f}%</strong> of credit limit.
                </div>
                <div class='driver-item'>
                    <strong>Repayment Ratio:</strong> Average monthly repayment ratio is <strong>{avg_pay_ratio*100:.1f}%</strong> of statement balance.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Actionable advisory block
            st.markdown("### 📣 Credit Committee Recommendation")
            if prob < 0.25:
                st.success("✅ **APPROVE CREDIT ACCESS:** The borrower exhibits exemplary payment metrics and negligible delinquency history. Standard interest terms and credit limit apply.")
            elif prob < 0.50:
                st.warning("⚠️ **APPROVE WITH CONDITIONS:** Borrower shows minor payment friction. It is recommended to apply a **20% limit buffer** and set up automated billing notifications.")
            else:
                st.error("❌ **DENY CREDIT APPLICATION:** Default risk is critical. Borrower has accumulated significant late payment history or elevated credit utilization. Deny credit or limit usage immediately.")
                
        else:
            # Default state before clicking assess
            st.info("ℹ️ Fill in the borrower information on the left and click **ASSESS CREDIT RISK** to compute default probabilities and view committee recommendations.")
            
            # Show live calculation of engineered features
            st.markdown("### 📊 Live Engineered Metrics Preview")
            st.write(f"- **Average Credit Utilisation (`AVG_UTIL_RATE`):** `{avg_util_rate*100:.1f}%` of Credit Limit")
            st.write(f"- **Average Repayment Consistency (`AVG_PAY_RATIO`):** `{avg_pay_ratio*100:.1f}%` of Outstanding Balance")
            st.write(f"- **Total Delinquent Months (`TOTAL_DELAY_MONTHS`):** `{total_delay_months}` months overdue")
            
            st.markdown("<small style='color:#6b7280;'>Features are computed instantly based on the inputs provided above.</small>", unsafe_allow_html=True)
