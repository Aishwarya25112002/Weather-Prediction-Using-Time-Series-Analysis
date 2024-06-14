import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from flask import Flask, request, render_template
import joblib
import warnings
import requests

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Load weather data
data = pd.read_csv('C:/Final Project/seattle-weather.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Split data into train and test sets
train_size = 0.8
train_data = data.iloc[:int(len(data) * train_size)]
test_data = data.iloc[int(len(data) * train_size):]

# Load trained ARIMA models
models_dir = 'models'
model1 = joblib.load(os.path.join(models_dir, 'model_1_ARIMA.pkl'))
model2 = joblib.load(os.path.join(models_dir, 'model_2_ARIMA.pkl'))
model3 = joblib.load(os.path.join(models_dir, 'model_3_ARIMA.pkl'))
model4 = joblib.load(os.path.join(models_dir, 'model_4_ARIMA.pkl'))

# Function to generate 7-day forecast
def generate_forecast(model, steps=7):
    forecast = model.forecast(steps=steps)
    return forecast

# OpenWeatherMap API key
API_KEY = '517a7d3bd6f272dcc460d6cb9974920f'  # Replace with your API key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_form', methods=['POST'])
def predict():
    city_name = request.form['city_name']
    api_key = API_KEY
    
    # Construct the weather URL
    weather_url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}'
    
    response = requests.get(weather_url)
    weather_data = response.json()
    
    if weather_data.get('cod') != 200:
        return render_template('index.html', prediction_text='City not found. Please enter a valid city name.')
    
    # Extract relevant weather information
    current_temp = weather_data['main']['temp'] - 273.15  # Convert from Kelvin to Celsius
    current_humidity = weather_data['main']['humidity']
    current_weather_desc = weather_data['weather'][0]['description']
    current_precipitation = weather_data.get('rain', {}).get('1h', 0)  # Precipitation in the last hour, default to 0 if not available
    current_wind = weather_data['wind']['speed']  # Wind speed in meters/second
    
    # Format temperature to two decimal places
    current_temp = f"{current_temp:.2f}"

    # Generate 7-day forecast
    precipitation_forecast = generate_forecast(model1)
    temp_max_forecast = generate_forecast(model2)
    temp_min_forecast = generate_forecast(model3)
    wind_forecast = generate_forecast(model4)
    
    # Format forecast output
    forecast_text = "7-day forecast:\n"
    for i in range(7):
        forecast_text += f"Day {i+1}: Precipitation: {precipitation_forecast[i]:.2f}mm, Max Temp: {temp_max_forecast[i]:.2f}°C, Min Temp: {temp_min_forecast[i]:.2f}°C, Wind Speed: {wind_forecast[i]:.2f} m/s\n"
    
    return render_template('index.html', 
                           prediction_text=f'Current weather in {city_name}: {current_weather_desc} with temperature {current_temp}°C, humidity {current_humidity}%, precipitation {current_precipitation}mm, and wind speed {current_wind} m/s',
                           forecast_text=forecast_text)

if __name__ == "__main__":
    from waitress import serve
    app.run(host='0.0.0.0', port=5051)    