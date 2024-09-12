import temperature_details as td
import time

def get_weather_data(location):
    while True:

        avg_temperature, avg_humidity, avg_sunlight_hours = td.get_average_weather(location)

        print(f"Average Temperature: {avg_temperature:.2f}°C")
        print(f"Average Humidity: {avg_humidity:.2f}%")
        print(f"Average Sunlight Hours: {avg_sunlight_hours:.2f} hours")

        return avg_temperature, avg_humidity, avg_sunlight_hours

        time.sleep(5)