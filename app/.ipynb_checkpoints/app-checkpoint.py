import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pathlib import Path # Used for robust path handling

# --- Configuration ---
# Use pathlib for robust path handling
MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'crime_predictor_model.joblib'
MODEL_PATH_STR = str(MODEL_PATH)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Violent Crime Risk Predictor (India)",
    page_icon="🚨",
    layout="wide"
)

@st.cache_resource
def load_model(path):
    """Load the trained model from the joblib file."""
    try:
        model = joblib.load(path) 
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. Please ensure the path and file name are correct and that you ran Notebook 2.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model globally
model = load_model(MODEL_PATH_STR) 

# --- Title and Description ---
st.markdown("""
    <div style="background-color:#ffebeb; padding:15px; border-radius:10px; text-align:center;">
        <h1 style="color:#cc0000;">🚨 Violent Crime Risk Predictor 🇮🇳</h1>
        <p style="color:#555555;">
            Predict the probability of a reported incident belonging to the **Violent Crime** domain
            based on key situational and geographical features.
        </p>
    </div>
    <br>
""", unsafe_allow_html=True)

# --- Feature Input Sidebars (Aesthetics) ---
col1, col2 = st.columns([1, 1])

# --- Input: Location and Time ---
with col1:
    st.subheader("Time and Location Context")
    
    # 1. Latitude and Longitude
    latitude = st.number_input(
        'Latitude ($\circ$)',
        min_value=-90.0, max_value=90.0, value=28.70, step=0.01,
        help="Geographic coordinate of the incident location (e.g., New Delhi: 28.70)"
    )
    longitude = st.number_input(
        'Longitude ($\circ$)',
        min_value=-180.0, max_value=180.0, value=77.10, step=0.01,
        help="Geographic coordinate of the incident location (e.g., New Delhi: 77.10)"
    )

    # 2. Time Features
    report_datetime = st.date_input(
        'Date of Incident Report',
        datetime.now().date(),
        help="Used to derive the month and day of the week."
    )
    report_time = st.time_input(
        'Time of Incident Report',
        datetime.now().time(),
        help="Used to derive the hour of the day (24-hour format)."
    )

    report_hour = report_time.hour
    report_day_of_week = report_datetime.weekday() # Monday is 0, Sunday is 6
    report_month = report_datetime.month

# --- Input: Victim and Incident Details ---
with col2:
    st.subheader("Victim and Incident Details")
    
    # 3. Victim Age and Police Deployed
    victim_age = st.slider(
        'Victim Age',
        min_value=0, max_value=100, value=35, step=1
    )
    police_deployed = st.number_input(
        'Police Deployed (Count)',
        min_value=1, max_value=50, value=15, step=1,
        help="The number of police officers involved or deployed at the scene."
    )
    
    # 4. Categorical Inputs
    # We will assume 'Female' was the dropped baseline for victim_gender.
    victim_gender = st.selectbox(
        'Victim Gender (Baseline assumed Female)',
        ['Male', 'Other', 'X', 'Female'], 
        index=3 # Default to Female (baseline)
    )
    
    # We will assume 'Blunt Object' or 'None' was the dropped baseline for weapon_used.
    weapon_used = st.selectbox(
        'Weapon Used',
        ['Blunt Object', 'Explosives', 'Firearm', 'Knife', 'None', 'Other', 'Poison', 'Unknown'], 
        index=2 # Default to Firearm, as it is now required
    )

    case_closed = st.radio(
        'Case Status (at time of prediction)',
        ['No', 'Yes'],
        index=0,
        horizontal=True,
        help="Is the case currently closed? We typically predict for open cases ('No')."
    )

# --- Prediction Logic ---

if st.button('Predict Risk', use_container_width=True):
    
    # 1. Create a dictionary of all features, ensuring 'weapon_used_Firearm' is now included.
    data = {
        # Continuous/Integer features
        'victim_age': [victim_age],
        'police_deployed': [police_deployed],
        'latitude': [latitude],
        'longitude': [longitude],
        'report_hour': [report_hour],
        'report_day_of_week': [report_day_of_week],
        'report_month': [report_month],
        'case_closed_Yes': [1 if case_closed == 'Yes' else 0],
        
        # --- VICTIM GENDER DUMMIES (M, X were kept) ---
        'victim_gender_M': [1 if victim_gender == 'Male' else 0],
        'victim_gender_X': [1 if victim_gender == 'X' else 0], 
        
        # --- WEAPON USED DUMMIES (Based on errors, we only include the categories that REMAINED in the model) ---
        'weapon_used_Firearm': [1 if weapon_used == 'Firearm' else 0], # <-- ADDED: REQUIRED BY ERROR
        'weapon_used_Explosives': [1 if weapon_used == 'Explosives' else 0],
        'weapon_used_Knife': [1 if weapon_used == 'Knife' else 0],
        'weapon_used_Other': [1 if weapon_used == 'Other' else 0],
        'weapon_used_Poison': [1 if weapon_used == 'Poison' else 0],
        'weapon_used_Unknown': [1 if weapon_used == 'Unknown' else 0], 
        
        # We continue to exclude the 5 features from previous errors:
        # 'crime_domain_Other Crime', 'crime_domain_Traffic Fatality', 'victim_gender_Other', 
        # 'weapon_used_Blunt Object', 'weapon_used_None'
    }

    # 2. Convert to DataFrame
    input_df = pd.DataFrame(data)

    # 3. Define the EXACT column order for the 16 remaining features (7 continuous + 1 case_closed + 2 victim_gender + 6 weapon_used)
    expected_columns = [
        'victim_age', 
        'police_deployed', 
        'latitude', 
        'longitude', 
        'report_hour',
        'report_day_of_week', 
        'report_month', 
        'victim_gender_M', 
        'victim_gender_X', 
        'weapon_used_Explosives', 
        'weapon_used_Firearm', # <-- ADDED: REQUIRED BY ERROR
        'weapon_used_Knife',
        'weapon_used_Other', 
        'weapon_used_Poison',
        'weapon_used_Unknown', 
        'case_closed_Yes'
    ]
    
    try:
        input_df = input_df[expected_columns]
    except KeyError as e:
        st.error(f"FATAL FEATURE MISMATCH: A column name is still missing or incorrect: {e}. The expected features list is wrong.")
        st.stop()


    # 4. Make Prediction
    try:
        # Get the probability of the positive class (Violent Crime, which is index 1)
        prediction_proba = model.predict_proba(input_df)[:, 1][0]
        prediction_score = round(prediction_proba * 100, 2)
        
        # 5. Display Result
        st.markdown("---")
        st.subheader("Prediction Result")
        
        # Dynamic Risk Assessment
        if prediction_score >= 70:
            risk_level = "HIGH RISK"
            color = "#ff4d4d"
        elif prediction_score >= 40:
            risk_level = "MODERATE RISK"
            color = "#ffc300"
        else:
            risk_level = "LOW RISK"
            color = "#32cd32"
            
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {color}1a; border: 2px solid {color};">
                <p style="font-size: 1.2em; color: {color}; text-align: center;">
                    Probability of **Violent Crime**:
                </p>
                <p style="font-size: 3em; font-weight: bold; color: {color}; text-align: center; margin: 0;">
                    {prediction_score}%
                </p>
                <p style="font-size: 1.5em; font-weight: bold; color: {color}; text-align: center;">
                    Overall Risk: {risk_level}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Footer/About ---
st.markdown("---")
st.caption("This application uses a Random Forest Classifier trained on sample Indian crime data (Project Version 3).")
