# config.py
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, AutoTokenizer
import torch


# config Hyperparameters
MODEL_ID = "facebook/wav2vec2-base-960h"
OUTPUT_DIR = "bachelorthesisModel"
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
MAX_STEPS = 2000
WARMUP_STEPS = 500
EVAL_STEPS = 1000
SAVE_STEPS = 1000
MAX_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the processor
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, '|': 4, 'E': 5, 'T': 6, 'A': 7, 'O': 8, 'N': 9, 'I': 10, 'H': 11, 'S': 12, 'R': 13, 'D': 14, 'L': 15, 'U': 16, 'M': 17, 'W': 18, 'C': 19, 'F': 20, 'G': 21, 'Y': 22, 'P': 23, 'B': 24, 'V': 25, 'K': 26, "'": 27, 'X': 28, 'J': 29, 'Q': 30, 'Z': 31}

def encode(transcription):
    transcription = transcription
    lettersMap = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, '|': 4, 'E': 5, 'T': 6, 'A': 7, 'O': 8, 'N': 9, 'I': 10, 'H': 11, 'S': 12, 'R': 13, 'D': 14, 'L': 15, 'U': 16, 'M': 17, 'W': 18, 'C': 19, 'F': 20, 'G': 21, 'Y': 22, 'P': 23, 'B': 24, 'V': 25, 'K': 26, "'": 27, 'X': 28, 'J': 29, 'Q': 30, 'Z': 31}
    # labels = [lettersMap[w] for w in transcription.split()]
    labels = [lettersMap[char] for char in transcription]
    return labels


def decode(labels):
    labels = labels
    lettersMap= { 0: '<pad>', 1: '<s>', 2: '</s>', 3: '<unk>', 4: ' ', 5: 'E', 6: 'T', 7: 'A', 8: 'O', 9: 'N', 10: 'I', 11: 'H', 12: 'S', 13: 'R', 14: 'D', 15: 'L', 16: 'U', 17: 'M', 18: 'W', 19: 'C', 20: 'F', 21: 'G', 22: 'Y', 23: 'P', 24: 'B', 25: 'V', 26: 'K', 27: "'", 28: 'X', 29: 'J', 30: 'Q', 31: 'Z'}
    transcription = ''.join(lettersMap[char] for char in labels)
    return transcription
# processor testing
'''
test_transcription = "WHAT|IS|MY|CURRENT|BANK|BALANCE"
test2_transcription = "WHAT IS MY CURRENT BANK BALANCE"
test3_transcription = "what is my current bank balance"
test4_transcription = test3_transcription.upper().replace(" ", "|")
encoded = processor(text=test_transcription)
decoded = processor.tokenizer.decode(encoded.input_ids)
encoded2 = processor(text=test2_transcription)
decoded2 = processor.tokenizer.decode(encoded2.input_ids)
encoded3 = processor(text=test3_transcription)
decoded3 = processor.tokenizer.decode(encoded3.input_ids)
encoded4 = processor(text=test4_transcription)
decoded4 = processor.tokenizer.decode(encoded4.input_ids)
print(f"Original transcription: {test_transcription}")
print(f"Encoded: {encoded.input_ids}")
print(f"Decoded: {decoded}")
print(f"Original transcription: {test2_transcription}")
print(f"Encoded: {encoded2.input_ids}")
print(f"Decoded: {decoded2}")
print(f"Original transcription: {test3_transcription}")
print(f"Encoded: {encoded3.input_ids}")
print(f"Decoded: {decoded3}")
print(f"Original transcription: {test4_transcription}")
print(f"Encoded: {encoded4.input_ids}")
print(f"Decoded: {decoded4}")
'''

'''transcription = ''.join(reverseLettersMap[label] for label in labels)
        transcription = transcription.replace('|', ' ')  # Replace '|' back to spaces'''