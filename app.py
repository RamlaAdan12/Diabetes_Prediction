# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model and Scaler with caching
@st.cache_resource
def load_models():
    try:
        model = joblib.load("random_forest_tuned.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except:
        try:
            model = joblib.load("random_forest_diabetes_model.pkl")
            scaler = joblib.load("scaler.pkl")
            return model, scaler
        except:
            st.error("No model files found! Please run retrain.py first.")
            st.stop()

model, scaler = load_models()

# Mappings
risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
bmi_mapping = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}

# Automatic calculation functions
def calculate_risk_level(glucose):
    if glucose >= 140:
        return "High"
    elif glucose >= 100:
        return "Medium"
    else:
        return "Low"

def calculate_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

REPORT_FILE = "diabetes_predictions.csv"

# Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "predictions_list" not in st.session_state:
    st.session_state.predictions_list = []

# Load/save functions
def load_predictions():
    if os.path.exists(REPORT_FILE):
        try:
            df = pd.read_csv(REPORT_FILE)
            st.session_state.predictions_list = df.to_dict('records')
        except:
            st.session_state.predictions_list = []

def save_prediction(record):
    df = pd.DataFrame(st.session_state.predictions_list)
    df.to_csv(REPORT_FILE, index=False)

load_predictions()

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2rem; color: #2c3e50; text-align: center; padding: 1rem;}
    .info-box {background-color: #e8f4f8; padding: 1rem; border-radius: 10px; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# Login Page
if not st.session_state.logged_in:
    st.markdown('<div class="main-header">Diabetes Prediction System</div>', unsafe_allow_html=True)
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("### Welcome!")
        st.markdown("Please login to continue")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Any password works")
        if st.button("Login", use_container_width=True):
            if username:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Please enter a username")

# Dashboard
else:
    st.markdown(f'<div class="main-header">Diabetes Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f"**Welcome, {st.session_state.username}!**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown(f"### User")
        st.markdown(f"**Name:** {st.session_state.username}")
        st.markdown(f"**Session Started:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.markdown("---")
        st.markdown("### Quick Stats")
        user_predictions = [p for p in st.session_state.predictions_list if p.get('Username') == st.session_state.username]
        st.metric("Your Predictions", len(user_predictions))
        if user_predictions:
            diabetic_count = len([p for p in user_predictions if p.get('Prediction') == 'Diabetic'])
            st.metric("Your Diabetic Predictions", diabetic_count)
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Tabs
    tabs = st.tabs(["New Prediction", "My Results", "Visualizations", "All Reports"])

    # TAB 1: NEW PREDICTION
    with tabs[0]:
        st.header("Enter Patient Details")
        st.markdown('<div class="info-box"><strong>Note:</strong> Risk Level and BMI Category will be calculated automatically based on your inputs!</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
            glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=30, step=1)
            insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=100, step=10)
        with col2:
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=28.0, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=35, step=1)

        # Auto calculate
        risk_level_label = calculate_risk_level(glucose)
        bmi_category_label = calculate_bmi_category(bmi)

        st.markdown("---")
        st.subheader("Automatically Calculated Values")
        st.write(f"Risk Level: {risk_level_label} (based on glucose: {glucose})")
        st.write(f"BMI Category: {bmi_category_label} (based on BMI: {bmi})")

        if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
            risk_level = risk_mapping[risk_level_label]
            bmi_category = bmi_mapping[bmi_category_label]
            input_df = pd.DataFrame([{
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age,
                'RiskLevel': risk_level,
                'BMICategory': bmi_category
            }])
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            st.session_state.prediction = pred
            st.session_state.proba = proba
            st.session_state.risk = risk_level_label
            st.session_state.bmi_cat = bmi_category_label
            st.session_state.prediction_done = True
            st.session_state.last_input = {
                'pregnancies': pregnancies,
                'glucose': glucose,
                'bmi': bmi,
                'age': age,
                'risk_level': risk_level_label,
                'bmi_category': bmi_category_label
            }
            record = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Username": st.session_state.username,
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age,
                "RiskLevel": risk_level_label,
                "BMICategory": bmi_category_label,
                "Prediction": "Diabetic" if pred == 1 else "Not Diabetic",
                "Probability_Not_Diabetic": round(proba[0], 4),
                "Probability_Diabetic": round(proba[1], 4),
                "Risk_Score": round(proba[1] * 100, 1)
            }
            st.session_state.predictions_list.append(record)
            save_prediction(record)
            st.success("Prediction complete! Check the Results tab.")

# ==================== TAB 2: MY RESULTS ====================
    with tabs[1]:
        st.header("Your Prediction Results")
        
        user_records = [p for p in st.session_state.predictions_list if p.get('Username') == st.session_state.username]
        
        if user_records:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", len(user_records))
            with col2:
                diabetic_count = len([p for p in user_records if p.get('Prediction') == 'Diabetic'])
                st.metric("Diabetic Cases", diabetic_count)
            with col3:
                avg_risk = np.mean([p.get('Risk_Score', 0) for p in user_records])
                st.metric("Average Risk Score", f"{avg_risk:.1f}%")
            with col4:
                st.metric("Most Recent", user_records[-1].get('Timestamp', 'N/A')[:10])
            
            st.markdown("---")
            st.subheader("Your Recent Predictions")
            
            display_df = pd.DataFrame(user_records[::-1])
            display_columns = ['Timestamp', 'Prediction', 'Risk_Score', 'Glucose', 'BMI', 'Age', 'Pregnancies', 'RiskLevel', 'BMICategory']
            display_df = display_df[display_columns]
            display_df.columns = ['Date/Time', 'Prediction', 'Risk Score %', 'Glucose', 'BMI', 'Age', 'Pregnancies', 'Risk Level', 'BMI Category']
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No predictions yet. Go to the New Prediction tab to make your first prediction!")
    
    # ==================== TAB 3: VISUALIZATIONS ====================
    with tabs[2]:
        st.header("Prediction Visualizations")
        
        if st.session_state.prediction_done:
            fig1 = go.Figure(go.Bar(
                x=["Not Diabetic", "Diabetic"],
                y=st.session_state.proba,
                marker_color=["#2ecc71", "#e74c3c"],
                text=[f"{st.session_state.proba[0]:.1%}", f"{st.session_state.proba[1]:.1%}"],
                textposition="auto"
            ))
            fig1.update_layout(
                title="Prediction Probability",
                yaxis=dict(range=[0, 1], title="Probability"),
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            risk_value = st.session_state.proba[1] * 100
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_value,
                title={'text': "Diabetes Risk Score"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': '#48dbfb'},
                        {'range': [30, 70], 'color': '#feca57'},
                        {'range': [70, 100], 'color': '#ff6b6b'}]
                }
            ))
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("Recommendations")
            if st.session_state.prediction == 1:
                st.error("""
                *Immediate Medical Consultation Recommended*
                - Schedule an appointment with a healthcare provider
                - Monitor blood glucose levels regularly
                - Consider lifestyle modifications
                - Discuss medication options with your doctor
                """)
            else:
                if st.session_state.proba[1] > 0.3:
                    st.warning("""
                    *Moderate Risk - Take Preventive Measures*
                    - Increase physical activity (30 min/day)
                    - Reduce sugar and refined carbohydrates
                    - Maintain healthy weight
                    - Regular check-ups every 6 months
                    """)
                else:
                    st.success("""
                    *Low Risk - Maintain Healthy Lifestyle*
                    - Continue healthy eating habits
                    - Regular exercise (150 min/week)
                    - Annual health screenings
                    """)
        else:
            st.info("ℹ️ Make a prediction first to see visualizations!")
    
    # ==================== TAB 4: ALL REPORTS ====================
    with tabs[3]:
        st.header("All Users Prediction Report")
        
        if st.session_state.predictions_list:
            all_df = pd.DataFrame(st.session_state.predictions_list)
            
            st.subheader("Overall Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", len(all_df))
            with col2:
                diabetic_total = len(all_df[all_df['Prediction'] == 'Diabetic'])
                st.metric("Total Diabetic Cases", diabetic_total)
            with col3:
                st.metric("Unique Users", all_df['Username'].nunique())
            with col4:
                avg_risk_all = all_df['Risk_Score'].mean()
                st.metric("Avg Risk Score", f"{avg_risk_all:.1f}%")
            
            st.markdown("---")
            st.subheader("Filter Reports")
            
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                users = ['All'] + sorted(all_df['Username'].astype(str).unique().tolist())
                selected_user = st.selectbox("Filter by User", users)
            
            with filter_col2:
                pred_options = ['All', 'Diabetic', 'Not Diabetic']
                selected_pred = st.selectbox("Filter by Prediction", pred_options)
            
            filtered_df = all_df.copy()
            if selected_user != 'All':
                filtered_df = filtered_df[filtered_df['Username'] == selected_user]
            if selected_pred != 'All':
                filtered_df = filtered_df[filtered_df['Prediction'] == selected_pred]
            
            st.caption(f"Showing {len(filtered_df)} records")
            
            display_cols = ['Timestamp', 'Username', 'Prediction', 'Risk_Score', 'Glucose', 'BMI', 'Age', 'Pregnancies', 'RiskLevel', 'BMICategory']
            st.dataframe(filtered_df[display_cols], use_container_width=True)
            
            st.markdown("---")
            st.subheader("Export Data")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download All Reports (CSV)",
                    data=all_df.to_csv(index=False),
                    file_name=f"diabetes_all_reports_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if len(filtered_df) > 0:
                    st.download_button(
                        label="Download Filtered Reports (CSV)",
                        data=filtered_df.to_csv(index=False),
                        file_name=f"diabetes_filtered_reports_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No predictions have been made yet.")
    
    st.markdown("---")
    st.caption("This tool is for educational purposes only. Always consult a healthcare provider for medical advice.")