import os
import csv
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from utils import check_device
from tqdm import tqdm

def compute_metrics(model, dataloader):
    device = check_device()
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_samples = 0
    total_correct = 0
    total_loss = 0.0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_loss += criterion(outputs, labels).item() * labels.size(0)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = total_correct / total_samples
    average_loss = total_loss / total_samples

    class_report = classification_report(
        true_labels, predicted_labels, output_dict=True)

    # return accuracy, average_loss, class_report
    save_metrics_to_csv(accuracy, average_loss, class_report)


def save_metrics_to_csv(accuracy, loss, class_report,experiment_name):
    os.makedirs(f'outputs/Indicator/{experiment_name}', exist_ok=True)
    filename = f'outputs/Indicator/{experiment_name}/metrics.csv'

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Accuracy', accuracy])
        writer.writerow(['Loss', loss])
        writer.writerow([])  # Empty row for separation
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score'])

        for label, metrics in class_report.items():
            if label != 'accuracy' and label != 'macro avg' and label != 'weighted avg':
                writer.writerow([label, metrics['precision'],
                                metrics['recall'], metrics['f1-score']])
