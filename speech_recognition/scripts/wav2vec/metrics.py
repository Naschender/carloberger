# metrics.py
from config import processor
import evaluate

# Initialize WER metric
wer_metric = evaluate.load("wer")
'''
def compute_metrics(outputs):
    preds = [output["preds"] for output in outputs]
    labels = [output["labels"] for output in outputs]

    # Decode predictions and labels
    pred_str = processor.batch_decode(preds, group_tokens=False)
    label_str = processor.batch_decode(labels, group_tokens=False)

    # Debug: Print predictions and references
    print("Predictions:", pred_str)
    print("References:", label_str)

    # Filter out empty references
    filtered_preds = []
    filtered_labels = []

    for pred, label in zip(pred_str, label_str):
        if label.strip() != "":
            filtered_preds.append(pred)
            filtered_labels.append(label)

    if not filtered_labels:
        print("Warning: All references are empty after filtering.")
        return 1.0  # Return a high WER score if all references are empty

    # Save predictions and references to a file
    with open("predictions_and_references.txt", "w", encoding="utf-8") as f:
        f.write("Predictions and References:\n")
        for pred, label in zip(filtered_preds, filtered_labels):
            f.write(f"Prediction: {pred}\n")
            f.write(f"Reference: {label}\n")
            f.write("\n")  # Add a blank line for readability

    # Calculate Word Error Rate (WER)
    wer = wer_metric.compute(predictions=filtered_preds, references=filtered_labels)
    
    # Debug: Print the computed WER
    print(f"Computed WER: {wer:.4f}")
    
    return wer

'''
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

        # Skip if predictions or references are empty
        if not decoded_preds or not decoded_labels:
            print("Skipping batch due to empty predictions or labels.")
            continue

        
        all_predictions.extend([decoded_preds.strip()])
        all_references.extend([decoded_labels.strip()])

    # Avoid divide-by-zero error
    if len(all_predictions) == 0 or len(all_references) == 0:
        print("No valid predictions or references for WER computation.")
        return float('inf')

    wer = wer_metric.compute(predictions=all_predictions, references=all_references)
    return wer
