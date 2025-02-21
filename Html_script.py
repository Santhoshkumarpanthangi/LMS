# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import streamlit as st

# Load the pre-trained model
model = pickle.load(open("C:/Users/ACT25/regression_model.pkl", 'rb'))

# Function to make predictions
def predict_air_quality(T, TM, Tm, SLP, H, VV, V, VM):
    # Prepare the input data
    input_data = np.array([[T, TM, Tm, SLP, H, VV, V, VM]])
    # Predict the PM2.5 (Air Quality)
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit App interface
st.title('Air Quality Prediction App')

st.write('### Enter the following details to predict the Air Quality (PM 2.5)')

# Input widgets for features
T = st.number_input('Average Annual Temperature (T)', min_value=-50.0, max_value=50.0, value=7.0)
TM = st.number_input('Annual Average Maximum Temperature (TM)', min_value=-50.0, max_value=50.0, value=9.0)
Tm = st.number_input('Average Annual Minimum Temperature (Tm)', min_value=-50.0, max_value=50.0, value=4.0)
SLP = st.number_input('Sea Level Pressure (SLP)', min_value=900.0, max_value=1100.0, value=1017.6)
H = st.number_input('Annual Average Humidity (H)', min_value=0.0, max_value=100.0, value=93.0)
VV = st.number_input('Annual Average Visibility (VV)', min_value=0.0, max_value=50.0, value=0.5)
V = st.number_input('Annual Average Wind Speed (V)', min_value=0.0, max_value=20.0, value=4.3)
VM = st.number_input('Annual Maximum Wind Speed (VM)', min_value=0.0, max_value=50.0, value=9.4)

# Button to get prediction
if st.button('Predict Air Quality'):
    prediction = predict_air_quality(T, TM, Tm, SLP, H, VV, V, VM)
    st.write(f'### Predicted PM2.5 (Air Quality): {prediction:.2f}')
    
    # Display some additional info or visuals (optional)
    st.write('### Model Evaluation:')
    st.write(f'Mean Absolute Error (MAE): 40.28')
    st.write(f'Mean Squared Error (MSE): 3057.66')
    st.write(f'Root Mean Squared Error (RMSE): 55.30')


    
