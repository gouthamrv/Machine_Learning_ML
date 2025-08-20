import numpy as np
import streamlit as st
import pickle

model = pickle.load(open('simple_linear_regression_model.pkl', 'rb'))

st.title('Salary Prediction App')

years_of_experience = st.number_input('Enter Years of Experience', min_value=0, max_value=50, value=0)
predicted_salary = model.predict([[years_of_experience]])

st.write(f'Predicted Salary for {years_of_experience} years of experience: ${predicted_salary[0]:,.2f}')
st.write('Model loaded successfully!') 
