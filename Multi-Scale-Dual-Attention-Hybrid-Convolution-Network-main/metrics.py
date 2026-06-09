from scipy.spatial.distance import directed_hausdorff
import numpy as np
""" Metrics ------------------------------------------ """
def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_instances = len(y_true)
    return correct_predictions / total_instances

def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):  #iou
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

## https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/319452
def hd_dist(preds, targets):
    haussdorf_dist = directed_hausdorff(preds, targets)[0]
    haussdorf_dist  = np.percentile(haussdorf_dist, 95)
    return haussdorf_dist

def iou(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def miou(y_true, y_pred):
    individual_iou = [iou(t, p) for t, p in zip(y_true, y_pred)]
    miou = np.mean(individual_iou)
    return miou


# def specificity(y_true, y_pred):
#     true_negative = ((1 - y_true) * (1 - y_pred)).sum()
#     false_positive = ((1 - y_true) * y_pred).sum()
#     return true_negative / (true_negative + false_positive + 1e-15)

