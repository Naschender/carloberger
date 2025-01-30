"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           ServerAPI.py
    Description:    This script contains the wav2vec2-model 2.0
                    custom server deployment source code using litserve package
    Author:         Carlo Berger, Aalen University
    Email:          Carlo.Berger@studmail.htw-aalen.de
    Created:        2024-11-15
    Last Modified:  2025-01-10
    Version:        1.0
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

import litserve as ls
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from config import MODEL_ID, processor

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# API Definition
class SpeechRecognitionAPI(ls.LitAPI):
    def setup(self, device):
        """
        Setup the model and processor. This method runs once at startup.
        """
        MODEL_PATH = "/home/tuancuong/workspaces/carloberger/speech_recognition/scripts/wav2vec/bachelorthesisModel_17-12-24"        
        
        print("Loading model and processor...")
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).to(DEVICE)
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        self.model.eval()

    def decode_request(self, request):
        """
        Decode the incoming request. Expects a file path as input.
        """
        if "file_path" not in request:
            raise ValueError("Request must include 'file_path'.")
        return request["file_path"]

    def preprocess_audio(self, audio_path):
        """
        Load and preprocess audio: resample to 16 kHz mono if needed.
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16 kHz mono
        if len(waveform.shape) > 1:
            waveform = torch.mean(waveform, dim=0)  # Convert stereo to mono

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(waveform)

        # Ensure waveform has shape [1, sequence_length]
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension

        return waveform


    def predict(self, audio_path):
        """
        Perform the transcription inference.
        """
        print(f"Processing audio file: {audio_path}")
        # Preprocess the audio
        waveform = self.preprocess_audio(audio_path)

        # Flatten waveform to 1D before passing to processor
        waveform = waveform.squeeze(0)  # Remove batch dimension if present
        print(f"Waveform shape after squeeze: {waveform.shape}")  # Debug

        # Pass waveform to the processor
        inputs = self.processor(
            waveform,  # Shape: [sequence_length]
            sampling_rate=16000,
            return_tensors="pt",
            padding=False
        )
        # inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=False)
        input_values = inputs.input_values.to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode transcription
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0], group_tokens=True)

        return transcription or "[No transcription could be generated]"

    def encode_response(self, output):
        """
        Convert the transcription output to the response payload.
        """
        return {"transcription": output}

# (STEP 2) - START THE SERVER
if __name__ == "__main__":
    server = ls.LitServer(SpeechRecognitionAPI(), accelerator="auto", max_batch_size=1)
    print("Starting Speech Recognition API server...")
    server.run(port=8000)
