# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 16:27:18 2025

@author: DELL
"""



import streamlit as st
import pandas as pd
import numpy as np
import joblib

model_path=joblib.load("C:/Users/DELL/Desktop/erp/lightgbm_model.joblib")
scaler_path=joblib.load("C:/Users/DELL/Desktop/erp/standardscaler_scaler.joblib")

st.title("Employee Retention Prediction")

# Input fields
enrollee_id=st.number_input("enrollee_id", min_value=1,step=10)
city_development_index = st.number_input("City Development Index", min_value=0.0, max_value=1.0, step=0.01)
gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
relevant_experience = st.selectbox("Relevant Experience", ['Has relevant experience', 'No relevant experience'])
enrolled_university = st.selectbox("Enrolled University", ['no_enrollment', 'Full time course', 'Part time course'])
education_level = st.selectbox("Education Level", ['Graduate', 'Masters', 'PhD'])
major_discipline = st.selectbox("Major Discipline", ['STEM', 'Business', 'Arts', 'Other'])
experience = st.number_input("Years of Experience", min_value=0.0, step=1.0)
company_size = st.selectbox("Company Size", ['<10', '10-50', '50-100', '100-500', '500-1000', '>1000'])
company_type = st.selectbox("Company Type", ['Pvt Ltd', 'Funded Startup', 'Public Sector', 'Early Stage Startup', 'Other'])
last_new_job = st.selectbox("Last New Job", ['1', '2', '3', '4', '>4', 'never'])
training_hours = st.number_input("Training Hours", min_value=0.0, step=1.0)

# Encode categorical values (simple encoding for demonstration)
encoded_gender = 1 if gender == 'Male' else 0
encoded_experience = 1 if relevant_experience == 'Has relevant experience' else 0
encoded_university = {'no_enrollment': 0, 'Full time course': 1, 'Part time course': 2}[enrolled_university]
encoded_education = {'Graduate': 0, 'Masters': 1, 'PhD': 2}[education_level]
encoded_major = {'STEM': 0, 'Business': 1, 'Arts': 2, 'Other': 3}[major_discipline]
encoded_company_size = {'<10': 0, '10-50': 1, '50-100': 2, '100-500': 3, '500-1000': 4, '>1000': 5}[company_size]
encoded_company_type = {'Pvt Ltd': 0, 'Funded Startup': 1, 'Public Sector': 2, 'Early Stage Startup': 3, 'Other': 4}[company_type]
encoded_last_new_job = {'1': 0, '2': 1, '3': 2, '4': 3, '>4': 4, 'never': 5}[last_new_job]

# Prepare input data
input_data = np.array([[enrollee_id,city_development_index, encoded_gender, encoded_experience, encoded_university,
                        encoded_education, encoded_major, encoded_company_size, encoded_company_type,
                        encoded_last_new_job, training_hours, experience]])
input_scaled = scaler_path.transform(input_data)

# Predict button
if st.button("Predict Retention"):
    prediction = model_path.predict(input_scaled)[0]
    if prediction == 0:
        st.success("The employee is likely to stay")
    else:
        st.error("The employee is likely to leave")






































# Input fields

city_development_index = st.number_input("City Development Index", min_value=0.0, max_value=1.0, step=0.01)
gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
relevant_experience = st.selectbox("Relevant Experience", ['Has relevant experience', 'No relevant experience'])
enrolled_university = st.selectbox("Enrolled University", ['no_enrollment', 'Full time course', 'Part time course'])
education_level = st.selectbox("Education Level", ['Graduate', 'Masters', 'PhD'])
major_discipline = st.selectbox("Major Discipline", ['STEM', 'Business', 'Arts', 'Other'])
experience = st.number_input("Years of Experience", min_value=0.0, step=1.0)
company_size = st.selectbox("Company Size", ['<10', '10-50', '50-100', '100-500', '500-1000', '>1000'])
company_type = st.selectbox("Company Type", ['Pvt Ltd', 'Funded Startup', 'Public Sector', 'Early Stage Startup', 'Other'])
last_new_job = st.selectbox("Last New Job", ['1', '2', '3', '4', '>4', 'never'])
training_hours = st.number_input("Training Hours", min_value=0.0, step=1.0)

# Encode categorical values (simple encoding for demonstration)
encoded_gender = 1 if gender == 'Male' else 0
encoded_experience = 1 if relevant_experience == 'Has relevant experience' else 0
encoded_university = {'no_enrollment': 0, 'Full time course': 1, 'Part time course': 2}[enrolled_university]
encoded_education = {'Graduate': 0, 'Masters': 1, 'PhD': 2}[education_level]
encoded_major = {'STEM': 0, 'Business': 1, 'Arts': 2, 'Other': 3}[major_discipline]
encoded_company_size = {'<10': 0, '10-50': 1, '50-100': 2, '100-500': 3, '500-1000': 4, '>1000': 5}[company_size]
encoded_company_type = {'Pvt Ltd': 0, 'Funded Startup': 1, 'Public Sector': 2, 'Early Stage Startup': 3, 'Other': 4}[company_type]
encoded_last_new_job = {'1': 0, '2': 1, '3': 2, '4': 3, '>4': 4, 'never': 5}[last_new_job]

# Prepare input data
input_data = np.array([[city_development_index, encoded_gender, encoded_experience, encoded_university,
                        encoded_education, encoded_major, encoded_company_size, encoded_company_type,
                        encoded_last_new_job, training_hours, experience]])
input_scaled = scaler_path.transform(input_data)

# Predict button
if st.button("Predict Retention"):
    prediction = model_path.predict(input_scaled)[0]
    if prediction == 0:
        st.success("The employee is likely to stay")
    else:
        st.error("The employee is likely to leave")