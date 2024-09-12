import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import base64
import os

# Suppress TensorFlow warnings (including TensorRT)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages

# Set the page title and favicon (plant logo)
st.set_page_config(page_title="AgriBazaar", page_icon="🌱")

# Ensure models are loaded only once to improve performance
@st.cache_resource
def load_model_and_transformers():
    try:
        # Load column transformer and scaler
        with open('column_transformer.pkl', 'rb') as f:
            ct = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            sc = pickle.load(f)

        # Load pre-trained TensorFlow model
        model = tf.keras.models.load_model('plant_growth_model.h5')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model, ct, sc
    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Please ensure the model and transformer files are uploaded correctly.")
        return None, None, None
    except Exception as e:
        st.error(f"Unexpected error while loading model or transformers: {e}")
        return None, None, None

# Load the model and transformers once
model, ct, sc = load_model_and_transformers()

# Function to add background animation using a farm/plant-based GIF
def add_bg_animation(file_path):
    try:
        # Read and encode local image/GIF
        with open(file_path, "rb") as f:
            data = f.read()
            encoded_image = base64.b64encode(data).decode()

        # Injecting the background image/GIF into CSS
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

            @media only screen and (max-width: 768px) {{
                .stApp {{
                    background-size: cover;
                }}
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
temperature = st.number_input('Temperature (तापमान °C)', min_value=-10.0, max_value=60.0, value=25.0)
humidity = st.number_input('Humidity (%) (नमी)', min_value=0.0, max_value=100.0, value=40.0)

# Mapping Hindi terms back to model-compatible values
soil_type_mapping = {
    'clay (मिट्टी)': 'clay',
    'sandy (रेतीली)': 'sandy',
    'loam (दोमट)': 'loam'
}

water_frequency_mapping = {
    'daily (रोजाना)': 'daily',
    'bi-weekly (सप्ताह में दो बार)': 'bi-weekly',
    'weekly (साप्ताहिक)': 'weekly'
}

fertilizer_type_mapping = {
    'none (कोई नहीं)': 'none',
    'chemical (रासायनिक)': 'chemical',
    'organic (जैविक)': 'organic'
}

# Error Handling for input values
if st.button("Predict (अनुमान लगाएं)"):
    try:
        # Convert input to English terms for the model
        soil_type_value = soil_type_mapping[soil_type]
        water_frequency_value = water_frequency_mapping[water_frequency]
        fertilizer_type_value = fertilizer_type_mapping[fertilizer_type]

        # Validate input ranges
        if not (0 <= sunlight_hours <= 24):
            st.error("Sunlight hours must be between 0 and 24.")
        elif not (-10 <= temperature <= 60):
            st.error("Temperature must be between -10°C and 50°C.")
        elif not (0 <= humidity <= 100):
            st.error("Humidity must be between 0% and 100%.")
        else:
            # Prepare input data
            input_data = [[soil_type_value, sunlight_hours, water_frequency_value, fertilizer_type_value, temperature, humidity]]

            # Ensure transformers and model are loaded correctly
            if ct is None or sc is None or model is None:
                st.error("Error loading the model or transformers. Please check your files.")
            else:
                # Transform and scale the input data
                input_data_transformed = ct.transform(input_data)
                input_data_scaled = sc.transform(input_data_transformed)

                # Make the prediction
                predicted_chance = model.predict(input_data_scaled)[0][0] * 100

                # Display the result
                st.success(f"The chance of crop growth is {predicted_chance:.2f}% (फसल वृद्धि की संभावना {predicted_chance:.2f}% है)")
    except KeyError as e:
        st.error(f"Key Error: {e}. Please ensure valid selections are made.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
