# save this as app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("random_forest_tuned.pkl")
scaler = joblib.load("scaler.pkl")

# Mapping for RiskLevel and BMICategory
risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
bmi_mapping = {"Underweight": 0, "Normal": 1, "Overweight/Obese": 2}

st.title("Diabetes Prediction UsinG RF")
st.header("Enter Patient Details")

# User inputs
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 30)
insulin = st.number_input("Insulin", 0, 800, 100)
bmi = st.number_input("BMI", 0.0, 70.0, 28.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 0, 120, 35)

# User-friendly dropdowns for categorical features
risk_level_label = st.selectbox("Risk Level", ["Low", "Medium", "High"])
bmi_category_label = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])

# Encode categorical inputs
risk_level = risk_mapping[risk_level_label]
bmi_category = bmi_mapping[bmi_category_label]

if st.button("Predict"):
    # Create DataFrame
    new_patient = pd.DataFrame([{
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
    
    # Scale features
    new_patient_scaled = scaler.transform(new_patient)
    
    # Make prediction
    pred = model.predict(new_patient_scaled)[0]
    proba = model.predict_proba(new_patient_scaled)[0]
    
    # Show results
    st.write(f"**Prediction:** {'Diabetic' if pred==1 else 'Not Diabetic'}")
    st.write(f"**Probability:** Not Diabetic = {proba[0]:.2f}, Diabetic = {proba[1]:.2f}")