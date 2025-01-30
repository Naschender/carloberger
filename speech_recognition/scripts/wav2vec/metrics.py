"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           metrics.py
    Description:    This script contains the wav2vec2-model 2.0
                    compute WER calculation function
    Author:         Carlo Berger, Aalen University
    Email:          Carlo.Berger@studmail.htw-aalen.de
    Created:        2024-12-10
    Last Modified:  2025-01-18
    Version:        3.0
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
from config import processor
import evaluate

# Initialize WER metric
wer_metric = evaluate.load("wer")

def compute_metrics(outputs):
    """
    Compute Word Error Rate (WER) from model outputs.

    Args:
    - outputs (list of dict): Each element contains keys `preds` and `labels`.

    Returns:
    - float: Word Error Rate (WER).
    """
    # Initialize empty lists for all predictions and references
    all_predictions = []
    all_references = []

    # Process each batch's outputs
    for output in outputs:
        # predictions and labels
        decoded_preds = output["preds"]
        decoded_labels = output["labels"]

        # Skip if references are empty
        if len(decoded_labels.strip()) == 0:
           print("Skipping batch due to empty predictions or labels.")
           continue

        
        all_predictions.extend([decoded_preds.strip()])
        all_references.extend([decoded_labels.strip()])

    # Avoid divide-by-zero error
    #if len(all_predictions) == 0 or len(all_references) == 0:
    #    print("No valid predictions or references for WER computation.")
    #    return float('inf')

    wer = wer_metric.compute(predictions=all_predictions, references=all_references)
    return wer
