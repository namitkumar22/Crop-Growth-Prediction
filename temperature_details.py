import requests
from datetime import datetime

# Function to get weather data from OpenWeatherMap
def get_weather_openweathermap(location, api_key):
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(base_url)
    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        
        # Calculate sunlight hours from sunrise and sunset
        sunrise = datetime.fromtimestamp(data['sys']['sunrise'])
        sunset = datetime.fromtimestamp(data['sys']['sunset'])
        sunlight_hours = (sunset - sunrise).seconds / 3600  # Convert seconds to hours

        return temperature, humidity, sunlight_hours
    else:
        print("Error with OpenWeatherMap API:", response.status_code)
        return None, None, None

# Function to get weather data from WeatherAPI
def get_weather_weatherapi(location, api_key):
    base_url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
    response = requests.get(base_url)
    if response.status_code == 200:
        data = response.json()
        temperature = data['current']['temp_c']
        humidity = data['current']['humidity']
        sunlight_hours = None  # WeatherAPI does not provide sunlight hours in current weather
        return temperature, humidity, sunlight_hours
    else:
        print("Error with WeatherAPI:", response.status_code)
        return None, None, None

# Function to get average weather data from OpenWeatherMap and WeatherAPI
def get_average_weather(location):
    # Your API keys for each weather service
    api_keys = {
        'openweathermap': 'a854f7654ced653fbce9fd5f0d7cc24d',
        'weatherapi': 'f281a42d51114d4986975713241209',
    }
    
    # Gather data from both sources
    temp_openweathermap, hum_openweathermap, sun_openweathermap = get_weather_openweathermap(location, api_keys['openweathermap'])
    temp_weatherapi, hum_weatherapi, sun_weatherapi = get_weather_weatherapi(location, api_keys['weatherapi'])

    # Filter out None values for valid data
    temperatures = [temp for temp in [temp_openweathermap, temp_weatherapi] if temp is not None]
    humidities = [hum for hum in [hum_openweathermap, hum_weatherapi] if hum is not None]
    sunlight_hours = [sun for sun in [sun_openweathermap] if sun is not None]  # Only OpenWeatherMap provides this

    # Calculate averages
    avg_temp = sum(temperatures) / len(temperatures) if temperatures else None
    avg_humidity = sum(humidities) / len(humidities) if humidities else None
    avg_sunlight_hours = sum(sunlight_hours) / len(sunlight_hours) if sunlight_hours else None

    return avg_temp, avg_humidity, avg_sunlight_hours

if __name__ == "__main__":
    location = input("Enter the city or village name: ")
    
    # Get average weather data
    avg_temperature, avg_humidity, avg_sunlight_hours = get_average_weather(location)

    print(f"Average Temperature: {avg_temperature:.2f}°C")
    print(f"Average Humidity: {avg_humidity:.2f}%")
    print(f"Average Sunlight Hours: {avg_sunlight_hours:.2f} hours")
