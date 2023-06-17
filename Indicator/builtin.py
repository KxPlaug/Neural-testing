import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
from utils import check_device
from tqdm import tqdm
import numpy as np
from Config.Dataset.coco_eval import CocoEvaluator
from Config.Dataset.coco_utils import get_coco_api_from_dataset
import torchvision
device = check_device()
from io import StringIO 
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def metric(model, dataloader, num_classes=1000):
    """_description_

    Args:
        model (_type_): pytorch model
        dataloader (_type_): your dataloader
    """

    losses = []
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
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


def ob_metric(model,dataloader):
    model.eval()
    iou_types = _get_iou_types(model)
    coco = get_coco_api_from_dataset(dataloader.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    for idx, (images, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images, targets)
        outputs = [{k: v.to("cpu") for k, v in t.items()}
                   for t in outputs]
        res = {target["image_id"].item(): output for target,
               output in zip(targets, outputs)}
        coco_evaluator.update(res)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    with Capturing() as output:
        coco_evaluator.summarize()
    return output