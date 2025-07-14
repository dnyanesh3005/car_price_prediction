import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('car_price_model.pkl')

st.title("🚗 Car Price Predictor")

st.markdown("Enter the car details below to predict the price.")

# Input features
sales = st.number_input("Sales (in thousands)", min_value=0.0)
resale = st.number_input("Resale value")
car_type = st.selectbox("Car Type (0 = Sedan, 1 = SUV)", [0, 1])
engine_s = st.number_input("Engine Size (liters)")
horsepow = st.number_input("Horsepower", min_value=50)
wheelbas = st.number_input("Wheelbase (inches)")
width = st.number_input("Width (inches)")
length = st.number_input("Length (inches)")
curb_wgt = st.number_input("Curb Weight (tons)")
fuel_cap = st.number_input("Fuel Capacity (gallons)")
mpg = st.number_input("Mileage (MPG)")
lnsales = st.number_input("Log(Sales)")

if st.button("Predict"):
    input_data = np.array([[sales, resale, car_type, engine_s, horsepow, wheelbas, width,
                            length, curb_wgt, fuel_cap, mpg, lnsales]])
    
    prediction = model.predict(input_data)
    st.success(f"Estimated Car Price: ₹ {round(prediction[0], 2)} lakh")
