"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           weather_plot.py
    Description:    This script receives the action to plot local weather,
                    therefore receives from JSON to panda converted data,
                     which should be plot and saved as plot locally
    Author:         Carlo Berger, Aalen University
    Email:          Carlo.Berger@studmail.htw-aalen.de
    Created:        2024-10-20
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

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

def plot_weather_data(df):
    # define or create plot directory
    output_dir = 'WeatherOutputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"actualWeather_{timestamp}.png")
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.set_xlabel('Date Time')
    ax1.set_ylabel('Temperature (C)', color='tab:red')
    ax1.plot(df['dt'], df['temp'], color='tab:red', label='Temperature')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Humidity (%)', color='tab:blue')
    ax2.plot(df['dt'], df['humidity'], color='tab:blue', label='Humidity')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    fig.tight_layout()
    plt.title('Temperature and Humidity over Time')
    
    # Save the plot as a file in the specified directory with the timestamp in the filename
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

