import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
from utils import check_device
from tqdm import tqdm
import numpy as np
import torchvision
device = check_device()
from io import StringIO 
import sys



def metric(model, dataloader, num_classes=1000):
    r"""
    metric
    Arguments:
        model (nn.Sequential): Black-box model being explained.
        dataloader (DataLoader): Dataloader.
        num_classes (int): Number of classes. (Default: 1000)
        
    Example::
        >>> metrics, class_report = metric(model, dataloader, num_classes=1000)
    """

    losses = []
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, loss = model.get_loss(images, labels)
            losses.append(loss.item())

            true_labels.append(labels.cpu().numpy())
            predicted_labels.append(outputs.cpu().numpy())

    average_loss = np.mean(losses)
    true_labels = np.concatenate(true_labels)
    predicted_labels = np.concatenate(predicted_labels)
    accuracy = np.mean(true_labels == predicted_labels.argmax(-1))
    cnf_confusion_matrix = confusion_matrix(
        true_labels, predicted_labels.argmax(-1))
    FP = cnf_confusion_matrix.sum(axis=0) - np.diag(cnf_confusion_matrix)
    TP = np.diag(cnf_confusion_matrix)
    FN = cnf_confusion_matrix.sum(axis=1) - np.diag(cnf_confusion_matrix)
    TN = cnf_confusion_matrix.sum() - (FP + FN + TP)

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    TPR = np.mean(TPR)
    TNR = np.mean(TNR)
    PPV = np.mean(PPV)
    NPV = np.mean(NPV)
    FPR = np.mean(FPR)
    FNR = np.mean(FNR)
    FDR = np.mean(FDR)
    if num_classes == 2:
        ROC_AUC = roc_auc_score(true_labels, predicted_labels[:, 1])
    else:
        ROC_AUC = roc_auc_score(true_labels, predicted_labels, multi_class='ovo')

    class_report = classification_report(
        true_labels, predicted_labels.argmax(-1), output_dict=True,digits=4,labels=list(range(num_classes)))

    metrics = {"Metric": ["Accuracy", "Loss", "TPR", "TNR", "PPV", "NPV", "FPR", "FNR", "FDR", "ROC_AUC"],
               "Value": [accuracy, average_loss, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ROC_AUC]}
    metrics = pd.DataFrame(metrics)
    metrics = metrics.round(4)
    
    class_report = pd.DataFrame(class_report).transpose()
    class_report = class_report.round(4)

    return metrics, class_report