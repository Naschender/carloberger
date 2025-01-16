import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torchaudio
from config import MODEL_ID, OUTPUT_DIR, BATCH_SIZE, LEARNING_RATE, MAX_STEPS, WARMUP_STEPS, EVAL_STEPS, SAVE_STEPS, MAX_EPOCHS, DEVICE, processor

def resample_audio(speech, original_sample_rate, target_sample_rate=16000):
    if original_sample_rate != target_sample_rate:
        print(f"Resampling audio from {original_sample_rate} Hz to {target_sample_rate} Hz.")
        speech = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=target_sample_rate
        )(torch.tensor(speech))
        speech = speech.numpy()  # Convert back to numpy array
    return speech

def transcribe(file_path):
    # Load processor and model
    # model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(DEVICE)
    model = Wav2Vec2ForCTC.from_pretrained('/home/tuancuong/workspaces/carloberger/speech_recognition/scripts/checkpoints/huggingface_model_17-12-24').to(DEVICE)
    # model = Wav2Vec2ForCTC.load_from_checkpoint('speech_recognition/scripts/checkpoints/model_epoch=5-val_loss=0.82.ckpt')

    model.eval()
    model.save_pretrained('/home/tuancuong/workspaces/carloberger/speech_recognition/scripts/wav2vec/bachelorthesisModel')
    # return 
    # Load the audio file
    speech, sample_rate = sf.read(file_path)

    # Resample if necessary
    speech = resample_audio(speech, sample_rate)

    # Preprocess the audio
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0], group_tokens=True)

    if not transcription:
        transcription = "[No transcription could be generated]"

    return transcription

if __name__ == "__main__":
    import sys
    '''
    if len(sys.argv) != 2:
        print("Usage: python test_wav2vec.py <audio_file_path>")
        sys.exit(1)
    '''

    # audio_path = sys.argv[1]
    audio_path = "speech_recognition/prediction_recordings/Aufzeichnung2.wav"
    result = transcribe(audio_path)
    print(f"Transcription: {result}")
