# main.py

import os
import subprocess

# Setze sicher, dass wir im richtigen Verzeichnis arbeiten
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Schritt 1: Bereite die Daten vor
print("Vorbereitung der Rohdaten...")
subprocess.run(['python3', 'scripts/prepare_data.py'], check=True)

# Schritt 2: Extrahiere Features
print("Extrahiere Features aus den Audiodaten...")
subprocess.run(['python3', 'scripts/feature_extraction.py'], check=True)

print("Datenaufbereitung abgeschlossen!")

# Schritt 3: Trainiere das Modell
print("Trainiere das Modell...")
subprocess.run(['python3', 'scripts/train_model.py'], check=True)

print("Pipeline abgeschlossen!")