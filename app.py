import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Load Model & Encoders
with open('Models/random_forest.pkl', 'rb') as file:
    model = pickle.load(file)

with open('Models/encoder_Peak_Season.pkl', 'rb') as file:
    encoder_Peak_Season = pickle.load(file)
with open('Models/encoder_Meal_Included.pkl', 'rb') as file:
    encoder_Meal_Included = pickle.load(file)
with open('Models/encoder_Ticket_Class.pkl', 'rb') as file:
    encoder_Ticket_Class = pickle.load(file)
with open('Models/encoder_Arrival_Time.pkl', 'rb') as file:
    encoder_Arrival_Time = pickle.load(file)
with open('Models/encoder_Departure_Time.pkl', 'rb') as file:
    encoder_Departure_Time = pickle.load(file)
with open('Models/OneHot_encoder.pkl', 'rb') as file:
    OneHot_encoder = pickle.load(file)

# Title
st.set_page_config(page_title="Airline Fare Prediction Model")
st.title("Airline Fare Prediction Model")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
        }
        .stSelectbox > label {
            font-weight: 500 !important;
        }
        .stNumberInput > label {
            font-weight: 500 !important;
        }
        .stSlider > label {
            font-weight: 500 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.markdown("#### Flight Details")

col1, col2 = st.columns(2)
with col1:
    Airline = st.selectbox('Airline', OneHot_encoder.categories_[0])
    Source = st.selectbox('Source Airport', OneHot_encoder.categories_[3])
    Day_of_Week = st.selectbox('Day of Week', OneHot_encoder.categories_[1])
with col2:
    Flight_Type = st.selectbox('Flight Type', OneHot_encoder.categories_[5])
    Destination = st.selectbox('Destination Airport', OneHot_encoder.categories_[4])
    Month = st.selectbox('Month', OneHot_encoder.categories_[2])

st.markdown("#### Date and Time Information")

col3, col4 = st.columns(2)
with col3:
    Journey_Date = st.date_input('Journey Date')
    Journey_Date_day = Journey_Date.strftime('%d')
    Departure_Time = st.selectbox('Departure Time', encoder_Departure_Time.categories_[0])
with col4:
    Arrival_Time = st.selectbox('Arrival Time', encoder_Arrival_Time.categories_[0])
    Duration_Hours = st.number_input('Flight Duration (in hours)', 0, 24, step=1)

st.markdown("#### Passenger and Booking Information")

col5, col6 = st.columns(2)
with col5:
    Booking_Days_Before = st.number_input('Booking Days Before', 1, 365, value=30)
    Ticket_Class = st.selectbox('Ticket Class', encoder_Ticket_Class.categories_[0])
    Stops = st.selectbox('Number of Stops', [0, 1, 2])
with col6:
    Fuel_Price_Impact = st.slider('Fuel Price Impact Multiplier', 0.5, 1.5, value=1.0)
    Baggage_Allowance_Kg = st.slider('Baggage Allowance (Kg)', 0, 30, value=20)
    Meal_Included = st.selectbox('Meal Included?', encoder_Meal_Included.categories_[0])

Peak_Season = st.selectbox('Peak Season?', encoder_Peak_Season.categories_[0])

st.markdown("---")

# Encode Inputs
Peak_Season_encoded = encoder_Peak_Season.transform([[Peak_Season]])[0][0]
Meal_Included_encoded = encoder_Meal_Included.transform([[Meal_Included]])[0][0]
Ticket_Class_encoded = encoder_Ticket_Class.transform([[Ticket_Class]])[0][0]
Arrival_Time_encoded = encoder_Arrival_Time.transform([[Arrival_Time]])[0][0]
Departure_Time_encoded = encoder_Departure_Time.transform([[Departure_Time]])[0][0]

categorical_inputs = [[Airline, Day_of_Week, Month, Source, Destination, Flight_Type]]
OneHot_encoded_array = OneHot_encoder.transform(categorical_inputs).toarray()
onehot_columns = OneHot_encoder.get_feature_names_out(['Airline', 'Day_of_Week', 'Month', 'Source', 'Destination', 'Flight_Type'])
onehot_df = pd.DataFrame(OneHot_encoded_array, columns=onehot_columns)

# Combine all features
input_data = pd.DataFrame({
    'Stops': [Stops],
    'Duration_Hours': [Duration_Hours],
    'Booking_Days_Before': [Booking_Days_Before],
    'Baggage_Allowance_Kg': [Baggage_Allowance_Kg],
    'Fuel_Price_Impact': [Fuel_Price_Impact],
    'Peak_Season': [Peak_Season_encoded],
    'Meal_Included': [Meal_Included_encoded],
    'Ticket_Class': [Ticket_Class_encoded],
    'Arrival_Time': [Arrival_Time_encoded],
    'Departure_Time': [Departure_Time_encoded],
    'Journey_Date': [Journey_Date_day]
})

input_final = pd.concat([input_data, onehot_df], axis=1)

# Align columns for Training
try:
    input_final = input_final[model.feature_names_in_]
except AttributeError:
    st.error("Model is missing 'feature_names_in_' attribute.")

# Prediction
st.markdown("#### Predict Fare")
if st.button('Predict'):
    prediction = model.predict(input_final)
    st.success(f"Estimated Flight Fare: ₹{prediction[0]:,.2f}")