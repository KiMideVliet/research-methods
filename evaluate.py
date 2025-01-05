import os
import gdown
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import pickle  # For loading saved predictions

    
def download_from_drive(link, output_path):
    """Download a file from Google Drive."""
    
    gdown.download(link, output_path, quiet=False)

#Load
def load_predictions(predictions_path):
    """Load saved predictions from Google Drive."""
    
    with open(predictions_path, "rb") as file:
        
        predictions = pickle.load(file)
        
    return predictions

#Calculate
def evaluate_metrics(predictions, ground_truths, class_names):
    """Evaluate precision, recall, and mAP per class."""
    
    metrics = {cls: {"precision": [], "recall": [], "AP": 0.0} for cls in class_names}

    for cls_idx, cls_name in enumerate(class_names):
        
        y_true = []
        
        y_scores = []
        
        for pred, gt in zip(predictions, ground_truths):
            
            gt_cls = [1 if obj["class"] == cls_idx else 0 for obj in gt]
            
            pred_cls = [d["conf"] for d in pred if d["class"] == cls_idx]

            y_true.extend(gt_cls)
            
            y_scores.extend(pred_cls)

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        ap = average_precision_score(y_true, y_scores)

        metrics[cls_name]["precision"] = precision
        
        metrics[cls_name]["recall"] = recall
        
        metrics[cls_name]["AP"] = ap

    return metrics

#Visualisation
def visualize_metrics(metrics):
    """Visualize precision-recall curves per class."""
    
    for cls, values in metrics.items():
        
        plt.figure()
        plt.plot(values["recall"], values["precision"], label=f'{cls} (AP={values["AP"]:.2f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve for {cls}")
        plt.legend()
        plt.grid()
        plt.show()

#Summary
def display_summary(metrics):
    """Print and summarize evaluation results."""
    
    print("\nEvaluation Summary\n")
    
    for cls, values in metrics.items():
        
        print(f"Class: {cls}")
        print(f"  AP: {values['AP']:.2f}")

#Evaluation
def model_evaluation(predictions_path, ground_truths, class_names):
    
    predictions = load_predictions(predictions_path)

    metrics = evaluate_metrics(predictions, ground_truths, class_names)

    visualize_metrics(metrics)
    
    display_summary(metrics)



if __name__ == "__main__":

    drivelink = 'https://drive.google.com/drive/folders/19nGwIZxLHHm_jCVff9CV1xS9gOfH1AsK?usp=drive_link'
    
    predictions_path = "saved_predictions.pkl"
    
    download_from_drive(drive_link, predictions_path)

    model_evaluation(predictions_path, ground_truths, class_names)

