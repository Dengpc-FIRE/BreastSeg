
import torch.nn as nn
import torch.nn.functional as F
import torch
#from medpy.metric.binary import dc,asd,hd,sensitivity, precision, ravd
import numpy as np
import sys
from scipy.ndimage import morphology
from medpy import metric

from torch.autograd import Variable
import math
sys.dont_write_bytecode = True  # don't generate the binray python file .pyc

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)



class DiceLoss(nn.Module):
    """
    define the dice loss
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1.
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1-((2. * intersection + smooth) / (A_sum + B_sum + smooth))

class Dice_Loss(nn.Module):
    def __init__(self, n_classes):
        super(Dice_Loss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


"""dice coefficient"""
def dice(pre, gt, tid=1):
    pre=pre==tid   #make it boolean
    gt=gt==tid     #make it boolean
    pre=np.asarray(pre).astype(np.bool_)
    gt=np.asarray(gt).astype(np.bool_)

    if pre.shape != gt.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have the same shape.")

    intersection = np.logical_and(pre, gt)
    dsc=(2. * intersection.sum() + 1e-07) / (pre.sum() + gt.sum() + 1e-07)

    return dsc

"""positive predictive value"""
def pospreval(pre,gt,tid=1):
    pre=pre==tid #make it boolean
    gt=gt==tid   #make it boolean
    pre=np.asarray(pre).astype(np.bool_)
    gt=np.asarray(gt).astype(np.bool_)

    if pre.shape != gt.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have the same shape.")

    intersection = np.logical_and(pre, gt)
    ppv=(1.0*intersection.sum() + 1e-07) / (pre.sum()+1e-07)

    return ppv

"""sensitivity"""
def sensitivity(pre,gt,tid=1):
    pre=pre==tid #make it boolean
    gt=gt==tid   #make it boolean
    pre=np.asarray(pre).astype(np.bool_)
    gt=np.asarray(gt).astype(np.bool_)

    if pre.shape != gt.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have the same shape.")

    intersection = np.logical_and(pre, gt)
    sen=(1.0*intersection.sum()+1e-07) / (gt.sum()+1e-07)

    return sen

"""specificity"""
def specificity(pre,gt):
    pre=pre==0 #make it boolean
    gt=gt==0   #make it boolean
    pre=np.asarray(pre).astype(np.bool)
    gt=np.asarray(gt).astype(np.bool)

    if pre.shape != gt.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have the same shape.")

    intersection = np.logical_and(pre, gt)
    spe=(1.0*intersection.sum()+1e-07) / (gt.sum()+1e-07)

    return spe

"""average surface distance"""#如何计算ASD相关的指标。
def surfd(pre, gt, tid=1, sampling=1, connectivity=1):
    pre=pre==tid   #make it boolean
    gt=gt==tid     #make it boolean

    if pre.shape != gt.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have the same shape.")

    input_1 = np.atleast_1d(pre.astype(np.bool_))
    input_2 = np.atleast_1d(gt.astype(np.bool_))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = np.logical_xor(input_1,morphology.binary_erosion(input_1, conn))
    Sprime = np.logical_xor(input_2,morphology.binary_erosion(input_2, conn))

    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)

    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
    return sds

def asd(pre, gt, tid=1, sampling=1, connectivity=1):
    sds = surfd(pre, gt, tid=tid, sampling=sampling, connectivity=connectivity)
    dis = sds.mean()
    return dis



def seg_metric(pre,gt):

    mask = (pre>0.5)
    gt = (gt>0.5)
    ASD = asd(mask, gt) #asd(mask, gt)
    # ASD = metric.binary.asd(mask, gt)
    DSC = dice(mask, gt)
    #HFD = dice(mask,gt)  #hd
    SEN = sensitivity(mask,gt)
    PPV = pospreval(mask,gt)
    #RAVD = dice(mask,gt)

    return DSC,PPV,SEN,ASD

def seg_metric2(pre,gt):
    mask = (pre > 0.5)
    gt = (gt > 0.5)

    ASD = metric.binary.asd(mask, gt)
    DSC = metric.binary.dc(mask, gt)
    # HFD = dice(mask,gt)  #hd
    SEN = metric.binary.sensitivity(mask, gt)
    PPV = metric.binary.positive_predictive_value(mask, gt)
    # RAVD = dice(mask,gt)

    return DSC, PPV, SEN, ASD

