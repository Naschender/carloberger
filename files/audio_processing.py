"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           audio_processing.py
    Description:    This script records a users voice command and sends it to
                    the transcription API
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
import json
import sounddevice as sd
import wave

def record_audio(file_name, duration=5, sample_rate=16000):
    print(sd.query_devices())
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished!")

    # Save as WAV file
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


def send_audio_to_api(api_url, audio_file):
    """
    Send a POST request with the audio file to the API endpoint.

    Parameters:
    api_url (str): The API endpoint URL.
    audio_file (str): The path to the audio file to be sent.

    Returns:
    str: The transcribed text from the API response.
    """
    # Prepare the payload
    payload = {
        "file_path": audio_file
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Send POST request
        response = requests.post(api_url, headers=headers, json=payload)
        '''
        # Debug request
        req = requests.Request('POST', api_url, headers=headers, json=payload)
        prepared = req.prepare()

        # Print the exact request being sent
        print(f"URL: {prepared.url}")
        print(f"Headers: {prepared.headers}")
        print(f"Body: {prepared.body}")
        # Send the request
        response = requests.Session().send(prepared)
        '''
        # Check the response status
        if response.status_code == 200:
            response_data = response.json()
            print("API Response:", response_data)
            return response_data.get("transcription", "")  # Extract and return the "text" field
        else:
            print(f"Error: {response.status_code}, Response: {response.text}")
            return ""

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while sending the request: {e}")
        return ""

