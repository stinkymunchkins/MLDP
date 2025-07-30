import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# loading the custom CSS styles to make the app look nice
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# setting up the page config (title, icon, and layout)
st.set_page_config(page_title="HDB Resale Price Predictor", page_icon="🏠", layout="wide")


model = joblib.load('rf_model.joblib')

# --- custom banner with gradient background ---
st.markdown("""
<div class="gradient-banner">
    <h1>🏠 HDB Resale Price Predictor</h1>
    <p>Get instant estimates for your HDB flat resale value with AI-powered predictions</p>
</div>
""", unsafe_allow_html=True)

# --- Property details section ---
st.markdown('<div class="section-title">🏡 Property Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)  # split the layout into two columns for input
with col1:
    # input for floor area of the flat
    floor_area = st.number_input(
        "Floor Area (sqm)", min_value=30, max_value=200, value=75,
        help="What is the size of your flat in square metres"
    )
    # dropdown to select flat type
    flat_type = st.selectbox(
        "Flat Type", ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"],
        help="Select your HDB flat type"
    )
with col2:
    # dropdown to select town where the flat is located
    town = st.selectbox(
        "Town", sorted([
            "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH", "BUKIT PANJANG",
            "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG", "CLEMENTI", "GEYLANG", "HOUGANG",
            "JURONG EAST", "JURONG WEST", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS",
            "PUNGGOL", "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
            "TOA PAYOH", "WOODLANDS", "YISHUN"
        ]),
        help="Select the town where your flat is located"
    )
    # dropdown to select flat model
    flat_model = st.selectbox(
        "Flat Model", sorted([
            "Improved", "New Generation", "Simplified", "Premium Apartment", "Maisonette", 
            "Apartment", "Adjoined flat", "Type S1", "Type S2", "Standard", "DBSS", "Terrace", 
            "Model A2", "2-room", "Type 1", "Type 2"
        ]),
        help="Select your flat model type"
    )

    # --- Show recent transactions for selected town in a dropdown ---
    try:
        # load historical data about HDB transactions
        df_hdb = pd.read_csv('hdb.csv')
        # Ensure 'month' column is parsed as datetime for easy sorting
        if 'month' in df_hdb.columns:
            df_hdb['month'] = pd.to_datetime(df_hdb['month'], errors='coerce')
            # Filter and show 5 most recent transactions for selected town
            recent = (
                df_hdb[df_hdb['town'] == town]
                .dropna(subset=['month'])
                .sort_values('month', ascending=False)
                .head(5)
            )
            if not recent.empty:
                with st.expander(f"🕵️‍♂️ Most Recent Transactions in {town}", expanded=False):
                    st.markdown(
                        "<div class='recent-transactions-title'>Most Recent HDB Transactions:</div>",
                        unsafe_allow_html=True
                    )
                    # process and display relevant data
                    recent_display = (
                        recent[['block', 'street_name', 'flat_model', 'flat_type', 'floor_area_sqm', 'resale_price']]
                        .rename(columns={
                            'block': 'Block',
                            'street_name': 'Street',
                            'flat_model': 'Model',
                            'flat_type': 'Type',
                            'floor_area_sqm': 'Area (sqm)',
                            'resale_price': 'Resale Price'
                        })
                        .reset_index(drop=True)
                    )
                    # format price and show data
                    recent_display['Resale Price'] = recent_display['Resale Price'].apply(lambda x: f"${int(x):,}")
                    st.dataframe(
                        recent_display,
                        hide_index=True,
                        use_container_width=True
                    )
            else:
                st.info(f"No recent transactions found for {town}.")
    except Exception as e:
        # show warning if there's an issue loading data
        st.warning(f"Could not load recent transactions data. ({e})")

st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)

# --- Age & Lease Information Section ---
st.markdown('<div class="section-title">📅 Age & Lease Information</div>', unsafe_allow_html=True)
# choosing the method of input for flat age
input_method = st.radio(
    "How would you like to input the age?",
    ["🗓️ Year Built", "⌨️ Manual Input"],
    horizontal=True,
    help="Choose your preferred method to specify the flat's age"
)
if input_method == "🗓️ Year Built":
    # get current year and calculate flat age
    current_year = datetime.now().year
    year_built = st.slider(
        "Year Built", min_value=1960, max_value=current_year, value=2000,
        help="What year was your flat built or when did you buy it?"
    )
    flat_age = current_year - year_built
    st.info(f"🕒 Calculated Flat Age: **{flat_age} years**")
    remaining_lease_years = max(1, 99 - flat_age)
    st.info(f"📋 Estimated Remaining Lease: **{remaining_lease_years} years**")
else:
    # manual input of flat age and remaining lease
    col_a, col_b = st.columns(2)
    with col_a:
        flat_age = st.number_input(
            "Flat Age (years)", min_value=0, max_value=60, value=25,
            help="How old is your flat?"
        )
    with col_b:
        remaining_lease_years = st.number_input(
            "Remaining Lease (years)", min_value=1, max_value=99, value=70,
            help="How many years left on the lease?"
        )

st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)

# --- prediction Button ---
if st.button("🔮 Predict My HDB Price", use_container_width=True):
    # gather input data for prediction
    input_df = pd.DataFrame({
        'floor_area_sqm': [floor_area],
        'flat_age': [flat_age],
        'remaining_lease_years': [remaining_lease_years],
        'flat_type': [flat_type],
        'town': [town],
        'flat_model': [flat_model]
    })
    
    # Reindex to match model feature order
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # make the prediction
    prediction = model.predict(input_df)[0]  # get the prediction
    st.markdown(f"""
    <div class="prediction-card">
        <h2>💰 Estimated Resale Price</h2>
        <div class="prediction-price">${prediction:,.0f}</div>
        <p style="font-size: 1.08rem; opacity: 0.92;">Based on current market analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # show property summary
    st.markdown(f"""
    <div class="summary-card">
        <h3>📊 Your Property Summary</h3>
        <div class="summary-details">
            🏠 <b>Flat Type:</b> {flat_type}<br>
            📐 <b>Floor Area:</b> {floor_area} sqm<br>
            🕒 <b>Age:</b> {flat_age} years<br>
            📍 <b>Location:</b> {town}<br>
            🏗️ <b>Model:</b> {flat_model}<br>
            📋 <b>Remaining Lease:</b> {remaining_lease_years} years
        </div>
    </div>
    """, unsafe_allow_html=True)

    # show warning about predictions
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Important Disclaimer</h4>
        <p>This is an Machine learned-powered predicted estimate based on historical data. Actual prices may vary due to:
        <ul>
            <li>🔨 Renovation status and interior condition</li>
            <li>🏢 Floor level and unit orientation</li>
            <li>🚇 Proximity to MRT, schools, and amenities</li>
            <li>✨ Unique property features and characteristics</li>
        </ul>
        <p style="margin-top:0.9em;">
            This prediction is just a guide, not an official valuation. For serious buying, selling, or financial decisions, always check with HDB or a licensed property agent.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- About, Tips, and Contact Section ---
st.markdown("""
<div class="about-section">
    <h3>🤖 About This Tool</h3>
    <p>
        Ever wondered how much your HDB flat might fetch on the resale market? This tool is here to help. It's like having a friendly property expert giving you a quick estimate based on key features agents look at.
    </p>
    <p>
        Whether you’re thinking of selling, dreaming of an upgrade, or just curious, this tool is built to make things a little easier. It uses machine learning trained on thousands of HDB transactions to give you an estimate that reflects real market trends.
    </p>
    <p>
        Share some basic details like type, size, and location, and in seconds, you’ll get an instant, data-driven estimate to help guide your plans.
    </p>
    <p>
        Please do remember that things like renovations or a great view can make a difference. So treat this estimate as a friendly guide, but always chat with HDB or a trusted agent for serious decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Footer with contact info ---
st.markdown("""
<div style="text-align:center; margin-top: 2.5rem; font-size:0.97rem; color: #888;">
    🏗️ Built for MLDP Project | Trained by Random Forest Machine Learning
</div>
""", unsafe_allow_html=True)
