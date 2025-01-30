"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           ConvertJSONtoPandas.py
    Description:    This script converts the JSON response from OpenWeatherAPI
                    to pandas for further processing and plotting
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

import pandas as pd

def convertJSONtoPandas(weather_json):
    weather_list = weather_json['list']
    weather_data = {
        'dt': [entry['dt'] for entry in weather_list],
        'temp': [entry['main']['temp'] for entry in weather_list],
        'humidity': [entry['main']['humidity'] for entry in weather_list],
        'pressure': [entry['main']['pressure'] for entry in weather_list]
    }
    
    # Create a DataFrame
    df = pd.DataFrame(weather_data)
    
    # Convert timestamps to datetime
    df['dt'] = pd.to_datetime(df['dt'], unit='s')
    
    return df
