"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           main.py
    Description:    This script starts the Project prototype and 
                    applies a straight forward pipeline from audio recording 
                    to excecution of the meant command
    Author:         Carlo Berger, Aalen University
    Email:          Carlo.Berger@studmail.htw-aalen.de
    Created:        2024-10-10
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

# import of the subfiles 
from audio_processing import record_audio, send_audio_to_api
from actions import execute_command
from predictCommand import predict

if __name__ == "__main__":
    # transcription API URL
    API_URL = "http://localhost:8000/predict"

    # temporary audio file for executing project prototype demonstration 
    AUDIO_FILE = "./recorded_audio.wav"

    # Debug with pre-recorded audio
    # AUDIO_FILE = "./Aufzeichnung3.wav"

    
    # Hauptlogik
    # Record Audio
    AUDIO_FILE =  record_audio(AUDIO_FILE, duration=5)  # Record 5 seconds of audio
    
    # transcribe audio into text via SpeechRecognition Model API
    text = send_audio_to_api(API_URL, AUDIO_FILE)
    # text = "PLEASE TURN ON THE LIGHTS AT THE WORKSTATION" # Debug text definition
    
    # Send transribed text to CLU model for action prediction
    prediction_result = predict(text)

    # Execution of predicted action
    if prediction_result:
        print("Prediction result:", prediction_result)
        top_intent = prediction_result['result']['prediction']['topIntent']
        execute_command(top_intent)
    else:
        print("Fehler bei der Vorhersage")


