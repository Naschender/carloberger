"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           localWheaterReq.py
    Description:    This script request the local weather data from OpenWeatherAPI
                    and processes it through plotting 
    Author:         Carlo Berger, Aalen University
    Email:          Carlo.Berger@studmail.htw-aalen.de
    Created:        2024-10-15
    Last Modified:  2025-01-30
    Version:        2.0
===============================================================================

    Copyright (c) 2025 Carlo Berger

    This software is provided "as is", without warranty of any kind, express
    or implied, including but not limited to the warranties of merchantability,
    fitness for a particular purpose, and non-infringement. In no event shall
    the authors or copyright holders be liable for any claim, damages, or other
    liability, whether in an action of contract, tort, or otherwise, arising
    from, out of, or in connection with the software or the use or other dealings
    in the software.

    All code is licenced under the opensource License. You may not use this file except
    in compliance with the License.

===============================================================================
"""

import requests
from ConvertJSONtoPandas import convertJSONtoPandas
from weather_plot import plot_weather_data  # Ensure weather_plot.py is in the same directory


def request_local_weather(google_api_key, openweathermap_api_key):
    # URL for the Google Geolocation API
    google_url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={google_api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "homeMobileCountryCode": 310,
        "homeMobileNetworkCode": 410,
        "radioType": "gsm",
        "carrier": "Vodafone",
        "considerIp": True
    }

    # Making the request to the Google Geolocation API
    response = requests.post(google_url, headers=headers, json=data)
    if response.status_code == 200:
        location_data = response.json()
        lat = location_data['location']['lat']
        lon = location_data['location']['lng']
        print(f"Location: Latitude = {lat}, Longitude = {lon}")

        # URL for the OpenWeatherMap Geocoding API
        geocoding_url = "http://api.openweathermap.org/geo/1.0/reverse"
        geocoding_params = {
            "lat": lat,
            "lon": lon,
            "limit": 1,  # Limit the number of results
            "appid": openweathermap_api_key
        }

        # Making the request to the OpenWeatherMap Geocoding API
        geocoding_response = requests.get(geocoding_url, params=geocoding_params)
        if geocoding_response.status_code == 200:
            geocoding_data = geocoding_response.json()
            print("Geocoding data:", geocoding_data)

            # URL for the OpenWeatherMap Weather API
            weather_url = f"http://api.openweathermap.org/data/2.5/forecast"
            weather_params = {
                "lat": lat,
                "lon": lon,
                "exclude": "minutely,hourly,daily,alerts",  # Adjustable parameters
                "appid": openweathermap_api_key,
                "units": "metric",  # Adjustable parameters
                "lang": "en"  # Language definition
            }

            # Making the request to the OpenWeatherMap Weather API
            weather_response = requests.get(weather_url, params=weather_params)
            if weather_response.status_code == 200:
                weather_data = weather_response.json()
                print("Weather data:", weather_data)

                # Convert JSON to Pandas DataFrame
                df = convertJSONtoPandas(weather_data)
                
                # Plot the data
                plot_weather_data(df)

            else:
                print(f"Weather request failed with status code: {weather_response.status_code}")
        else:
            print(f"Geocoding request failed with status code: {geocoding_response.status_code}")
    else:
        print(f"Geolocation request failed with status code: {response.status_code}")