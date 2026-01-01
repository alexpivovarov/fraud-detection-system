'''
Streamlit Dashboard for Fraud Detection
'''
import streamlit as st
import requests
import json

# API endpoint
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Fraud Detection System")
st.markdown("Real-time transaction fraud scoring powered by XGBoost")

# Sidebar for input
st.sidebar.header("Transaction Details")

transaction_id = st.sidebar.number_input("Transaction ID", min_value=1, value=1001)
transaction_amt = st.sidebar.number_input("Amount ($)", min_value=0.01 ,value=150.00, step=10.0)
card1 = st.sidebar.number_input("Card ID", min_value=1, value=13926)

card4 = st.sidebar.selectbox("Card Network", ["visa", "mastercard", "american express", "discover", None])
card6 = st.sidebar.selectbox("Card type", ["debit", "credit", None])

email_domain = st.sidebar.selectbox(
    "Email Domain",
    ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com", None]
)

product_cd = st.sidebar.selectbox("Product code", ["W", "C", "R", "H", "S"])
device_type = st.sidebar.selectbox("Device Type", ["desktop", "mobile", None])

# Submit button 
if st.sidebar.button(" Check for Fraud", type="primary"):

    # Build transaction payload
    transaction = {
        "TransactionID": transaction_id,
        "TransactionAmt": transaction_amt,
        "card1": card1,
        "card4": card4,
        "card6": card6,
        "P_emaildomain": email_domain,
        "ProductCD": product_cd,
        "DeviceType": device_type
    }


# Call API
    try:
        response = requests.post(f"{API_URL}/predict", json=transaction)

        if response.status_code == 200:
            result = response.json()

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")

            with col2:
                if result['is_fraud']:
                    st.error(f"Risk level: {result['risk_level']}")
                else:
                    st.success(f"Risk Level: {result['risk_level']}")

            with col3:
                st.metric("Threshold", f"{result['threshold']:.0%}")

            # Fraud alert banner
            if result['is_fraud']:
                st.error("FRAUD ALERT: This transaction is flagged as potentially fraudulent")
            else:
                st.success("Transaction appears legitimate")

            # Details
            st.subheader("Transaction Details")
            st.json(transaction)

            # Response
            st.subheader("Model response")
            st.json(result)

        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(f" Cannot connect to API. Make sure the API is running: 'python3 api/fraud_api.py")

# Health check
st.sidebar.markdown("---")
st.sidebar.subheader("System status")

try:
    health = requests.get(f"{API_URL}/health").json()
    if health['predictor_loaded']:
        st.sidebar.success("Model loaded")
    else:
        st.sidebar.error("Model not loaded")

    if health['redis_connected']:
        st.sidebar.success("✓ Redis connected")
    else:
        st.sidebar.warning("⚠ Redis not connected")
except:
    st.sidebar.error("✗ API offline")


st.sidebar.markdown("---")

