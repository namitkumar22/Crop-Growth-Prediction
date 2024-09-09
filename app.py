import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import base64
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import regularizers

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the page title and favicon
st.set_page_config(page_title="AgriBazaar", page_icon="🌱")

# Function to add background animation
def add_bg_animation(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            encoded_image = base64.b64encode(data).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/gif;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                height: 100vh;
                width: 100vw;
            }}
            .stTextInput, .stSelectbox, .stNumberInput {{
                background-color: rgba(0,0,0,0.4);
                border-radius: 10px;
                padding: 10px;
            }}
            .stButton>button {{
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError as e:
        st.error(f"Background animation file not found: {e}.")
    except Exception as e:
        st.error(f"Error loading background animation: {e}")

# Call the function to add background animation (change to a valid path)
add_bg_animation("1.png")

# Title and description
st.title("🌱 AgriBazaar: Plant Growth Prediction (फसल वृद्धि की भविष्यवाणी)")
st.write("Enter the details below to predict the chances of crop growth: (नीचे जानकारी दर्ज करें ताकि फसल वृद्धि की संभावना का अनुमान लगाया जा सके)")

# Input fields for prediction with Hindi translations
soil_type = st.selectbox('Soil Type (मिट्टी का प्रकार)', ['clay (मिट्टी)', 'sandy (रेतीली)', 'loam (दोमट)'])
sunlight_hours = st.number_input('Sunlight Hours (सूरज की रोशनी के घंटे)', min_value=0.0, max_value=24.0, value=8.0)
water_frequency = st.selectbox('Water Frequency (पानी देने की आवृत्ति)', ['daily (रोजाना)', 'bi-weekly (सप्ताह में दो बार)', 'weekly (साप्ताहिक)'])
fertilizer_type = st.selectbox('Fertilizer Type (उर्वरक प्रकार)', ['none (कोई नहीं)', 'chemical (रासायनिक)', 'organic (जैविक)'])
temperature = st.number_input('Temperature (तापमान °C)', min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input('Humidity (%) (नमी)', min_value=0.0, max_value=100.0, value=40.0)

# Mapping Hindi terms back to model-compatible values
soil_type_mapping = {'clay (मिट्टी)': 'clay', 'sandy (रेतीली)': 'sandy', 'loam (दोमट)': 'loam'}
water_frequency_mapping = {'daily (रोजाना)': 'daily', 'bi-weekly (सप्ताह में दो बार)': 'bi-weekly', 'weekly (साप्ताहिक)': 'weekly'}
fertilizer_type_mapping = {'none (कोई नहीं)': 'none', 'chemical (रासायनिक)': 'chemical', 'organic (जैविक)': 'organic'}

# Create and compile the model
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize model
model = create_model()

# Error Handling for input values
if st.button("Predict (अनुमान लगाएं)"):
    try:
        soil_type_value = soil_type_mapping[soil_type]
        water_frequency_value = water_frequency_mapping[water_frequency]
        fertilizer_type_value = fertilizer_type_mapping[fertilizer_type]

        # Prepare input data
        input_data = [[soil_type_value, sunlight_hours, water_frequency_value, fertilizer_type_value, temperature, humidity]]

        # Encode categorical features and scale numerical features
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 2, 3])], remainder='passthrough')
        input_data_transformed = ct.fit_transform(input_data)
        sc = StandardScaler()
        input_data_scaled = sc.fit_transform(input_data_transformed)

        # Make the prediction
        predicted_chance = model.predict(input_data_scaled)[0][0] * 100

        # Display the result
        st.success(f"The chance of crop growth is {predicted_chance:.2f}% (फसल वृद्धि की संभावना {predicted_chance:.2f}% है)")
    except KeyError as e:
        st.error(f"Key Error: {e}. Please ensure valid selections are made.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
