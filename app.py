import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import base64
import os
import weather_data as wd
from streamlit_extras.let_it_rain import rain

# Suppress TensorFlow warnings (including TensorRT)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the page title and favicon (plant logo)
st.set_page_config(page_title="AgriBazaar", page_icon="🌱")

# CSS to remove the Streamlit menu and footer
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

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
        st.error(f"File not found: {e}. Ensure files are uploaded correctly. (फ़ाइल नहीं मिली: {e}. कृपया सुनिश्चित करें कि फ़ाइलें सही ढंग से अपलोड की गई हैं।)")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model or transformers: {e} (मॉडल या ट्रांसफार्मर्स लोड करते समय त्रुटि: {e})")
        return None, None, None

# Load the model and transformers
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

st.title("🌱 AgriBazaar: Plant Growth Prediction (फसल वृद्धि भविष्यवाणी)")
st.write("Enter the details below to predict crop growth chances (फसल वृद्धि की संभावना जानने के लिए विवरण भरें)")

# Input fields
person_location = st.text_input("Your Location (आपका स्थान) [For fetching weather info (मौसम जानकारी के लिए)]", placeholder="Any village, district, city name (कोई गाँव, जिला, शहर का नाम)").lower()

if person_location:
    try:
        temperature_auto, humidity_auto, sunlight_hours_auto = wd.get_weather_data(person_location)

        # Show input fields only after weather data is successfully fetched
        temperature = st.number_input('Temperature (तापमान °C)', min_value=-10.0, max_value=60.0, value=temperature_auto, help="Automatically fetched (स्वतः प्राप्त)")
        humidity = st.number_input('Humidity (%) (नमी)', min_value=0.0, max_value=100.0, value=humidity_auto, help="Automatically fetched (स्वतः प्राप्त)")
        sunlight_hours = st.number_input('Sunlight Hours (सूरज की रोशनी के घंटे)', min_value=0.0, max_value=24.0, value=sunlight_hours_auto, help="Automatically fetched (स्वतः प्राप्त)")

        # Select boxes with initial empty value
        soil_type = st.selectbox('Soil Type (मिट्टी का प्रकार)', ['', 'clay (मिट्टी)', 'sandy (रेतीली)', 'loam (दोमट)'])
        water_frequency = st.selectbox('Water Frequency (पानी देने की आवृत्ति)', ['', 'daily (रोजाना)', 'bi-weekly (सप्ताह में दो बार)', 'weekly (साप्ताहिक)'])
        fertilizer_type = st.selectbox('Fertilizer Type (उर्वरक प्रकार)', ['', 'none (कोई नहीं)', 'chemical (रासायनिक)', 'organic (जैविक)'])

        # Mappings for input
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

        # Check if required inputs are provided before processing
        if st.button('Predict Growth Chance (फसल वृद्धि की संभावना बताएं)'):
            if not person_location:
                st.error("Location is required for fetching weather data. (मौसम जानकारी के लिए स्थान भरना आवश्यक है।)")
            elif soil_type == '' or water_frequency == '' or fertilizer_type == '':
                st.error("Please fill in all the required fields. (कृपया सभी आवश्यक फ़ील्ड भरें।)")
            else:
                try:
                    # Convert selected values for the model
                    soil_type_value = soil_type_mapping.get(soil_type)
                    water_frequency_value = water_frequency_mapping.get(water_frequency)
                    fertilizer_type_value = fertilizer_type_mapping.get(fertilizer_type)

                    # Prepare input data
                    input_data = [[soil_type_value, sunlight_hours, water_frequency_value, fertilizer_type_value, temperature, humidity]]

                    # Transform and scale input data
                    input_data_transformed = ct.transform(input_data)
                    input_data_scaled = sc.transform(input_data_transformed)

                    # Make prediction
                    predicted_chance = model.predict(input_data_scaled)[0][0] * 100

                    # Display result
                    st.success(f"The chance of crop growth is {predicted_chance:.2f}% (फसल वृद्धि की संभावना {predicted_chance:.2f}% है)")
                except Exception as e:
                    st.error(f"Error during prediction: {e} (भविष्यवाणी करते समय त्रुटि: {e})")
    except Exception as e:
        st.info(f"Please Enter a valid location or any nearest city (कृपया एक सही स्थान या नजदीकी शहर का नाम डालें)")
else:
    st.info("Please enter a location to fetch weather information. (मौसम जानकारी प्राप्त करने के लिए कृपया एक स्थान भरें।)")
