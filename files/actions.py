"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           actions.py
    Description:    This script stores the implemented actions of the prototype
    Author:         Carlo Berger, Aalen University
    Email:          Carlo.Berger@studmail.htw-aalen.de
    Created:        2024-10-12
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
from localWheaterReq import request_local_weather


# Hyperparameters
GOOGLE_API_KEY = "***"
OPENWEATHERMAP_API_KEY = "***"

TURN_ON_LIGHT_URL = "https://www.homeassistant.com/api/webhook/-TURN_ON_LIGHT"
TURN_OFF_LIGHT_URL = "https://www.homeassistant.com/api/webhook/-TURN_OFF_LIGHT"
TURN_ON_VENT_URL = "https://www.homeassistant.com/api/webhook/-TURN_ON_VENT"
TURN_OFF_VENT_URL = "https://www.homeassistant.com/api/webhook/-TURN_OFF_VENT"
TURN_ON_WORK_LIGHT_URL = "https://www.homeassistant.com/api/webhook/-TURN_ON_WORK_LIGHT"
TURN_OFF_WORK_LIGHT_URL = "https://www.homeassistant.com/api/webhook/-TURN_OFF_WORK_LIGHT_URL"
# call of HomeAssitstant Webhooks
def turn_on_light():
    url = TURN_ON_LIGHT_URL
    response = requests.post(url)
    print(f"Turn on light: {response.status_code}")
    print(f"Debug: Light turned on response - {response.text}")

def turn_off_light():
    url = TURN_OFF_LIGHT_URL
    response = requests.post(url)
    print(f"Turn off light: {response.status_code}")
    print(f"Debug: Light turned off response - {response.text}")

def turn_on_vent():
    url = TURN_ON_VENT_URL
    response = requests.post(url)
    print(f"Turn on ventilator: {response.status_code}")
    print(f"Debug: Ventilator turned on response - {response.text}")

def turn_off_vent():
    url = TURN_OFF_VENT_URL
    response = requests.post(url)
    print(f"Turn off ventilator: {response.status_code}")
    print(f"Debug: Ventilator turned off response - {response.text}")

def turn_on_WorkstationLight():
    url = TURN_ON_WORK_LIGHT_URL
    response = requests.post(url)
    print(f"Turn on Workstation Light: {response.status_code}")
    print(f"Debug: Workstation Light turned on response - {response.text}")

def turn_off_WorkstationLight():
    url = TURN_OFF_WORK_LIGHT_URL
    response = requests.post(url)
    print(f"Turn off Workstation Light: {response.status_code}")
    print(f"Debug: Workstation Light turned off response - {response.text}")

# commandset definition
def execute_command(command):
    print(f"Debug: Received command - {command}")
    
    if command == "Turn the lights off":
        print("Executing: Turn the lights off")
        turn_off_light()
    elif command == "Turn the lights on":
        print("Executing: Turn the lights on")
        turn_on_light()
    elif command == "Turn the Ventilator on":
        print("Executing: Turn the ventilator on")
        turn_on_vent()
    elif command == "Turn the Ventilator off":
        print("Executing: Turn the ventilator off")
        turn_off_vent()
    elif command == "Gather weather information":
        print("Executing: Make a weather request")
        google_api_key = GOOGLE_API_KEY
        openweathermap_api_key = OPENWEATHERMAP_API_KEY
        request_local_weather(google_api_key, openweathermap_api_key)
        print("Debug: Weather request completed")
    elif command == "Turn Workstation lights on":
        print("Executing: Turn Workstation Light on")
        turn_on_WorkstationLight()
    elif command == "Turn Workstation lights off":
        print("Executing: Turn Workstation Light off")
        turn_off_WorkstationLight()
    elif command == "Test Answer":
        print("You are testing the Bachelorthesis of Carlo Berger")
    else:
        print("Unknown command:", command)
