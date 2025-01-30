"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           predictCommand.py
    Description:    This script sends the transcribed text to CLU API
                    for predicting the most likely action and returns 
                    this most likely prediction
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

# prediction URL
prediction_url = "https://projectname.cognitiveservices.azure.com/language/model-parameters"

# API key
api_key = "API_ID_HERE"

# Apim Request Id
apim_request_id = "APIM_ID_HERE"

# participant Id
participant_id = "PARTICIPANT_ID_HERE"

# Query Language
query_language = "en"

# Project name
project_name = "BachelorThesis***"

# deployment name
deployment_name = "BasicModel***"

def predict(input_data):
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Apim-Request-Id": apim_request_id,
        "Content-Type": "application/json"
    }

    data = {
        "kind": "Conversation",
        "analysisInput": {
            "conversationItem": {
                "id": participant_id,
                "text": input_data,
                "modality": "text",
                "language": query_language,
                "participantId": participant_id
            }
        },
        "parameters": {
            "projectName": project_name,
            "verbose": True,
            "deploymentName": deployment_name,
            "stringIndexType": "TextElement_V8"
        }
    }

    response = requests.post(prediction_url, headers=headers, json=data)
    
    '''
    # Debug request
    req = requests.Request('POST', prediction_url, headers=headers, json=data)
    prepared = req.prepare()

    # Print the exact request being sent
    print(f"URL: {prepared.url}")
    print(f"Headers: {prepared.headers}")
    print(f"Body: {prepared.body}")
    # Send the request
    response = requests.Session().send(prepared)
    '''
    
    if response.status_code == 200:
        result = response.json()
        # return the result
        return result
    else:
        print(f"Prediction request failed with status code {response.status_code}")
        return None
