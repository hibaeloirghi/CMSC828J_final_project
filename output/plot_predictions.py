import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Define the function to process the JSONL file and compute metrics
def generate_confusion_matrix_and_accuracy(file_path, output_image_path='confusion_matrix.png'):
    # Initialize lists for labels and predicted labels
    labels = []
    predicted_labels = []

    # Read the JSONL file and parse the data
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            labels.append(data['label'])
            predicted_labels.append(data['predicted_label'])
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predicted_labels)
    accuracy = accuracy_score(labels, predicted_labels)
    
    # Print accuracy
    print(f"Accuracy: {accuracy:.2%}")

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=set(labels), yticklabels=set(labels)) # , cmap='Blues'
    plt.xlabel('Predicted Label')
    plt.ylabel('XNLI Label')
    plt.title('Confusion Matrix for Llama2-7b XNLI classification in Arabic')

    # Save the plot as an image
    plt.savefig('/fs/clip-scratch/eloirghi/CMSC828J_final_project/output/predict_label_ar_llama2_7b_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix heatmap saved as {'/fs/clip-scratch/eloirghi/CMSC828J_final_project/output/predict_label_ar_llama2_7b_confusion_matrix.png'}")

file_path = '/fs/clip-scratch/eloirghi/CMSC828J_final_project/output/predict_label_ar_llama2_7b_3389728.jsonl'
generate_confusion_matrix_and_accuracy(file_path)