import torch
import torch.nn as nn
import torch.nn.functional as F

""" Loss Functions -------------------------------------- """
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)
        loss = self.bceloss(pred_flat, target_flat)
        return loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss

"""BCE + DICE Loss"""
# class DiceBCELoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceBCELoss, self).__init__()
#
#     def forward(self, inputs, targets, smooth=1):
#         inputs = torch.sigmoid(inputs)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         intersection = (inputs * targets).sum()
#         dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         Dice_BCE = BCE + dice_loss
#         return Dice_BCE

class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6], n_labels=1):
        super(WeightedBCE, self).__init__()
        self.weights = weights
        self.n_labels = n_labels

    def forward(self, logit_pixel, truth_pixel):
        # print("====",logit_pixel.size())
        if self.n_labels == 1:
            logit = logit_pixel.view(-1)
            truth = truth_pixel.view(-1)
            assert (logit.shape == truth.shape)
            loss = F.binary_cross_entropy(logit, truth, reduction='none')
            pos = (truth > 0.5).float()
            neg = (truth < 0.5).float()

            pos_weight = pos.sum().item() + 1e-12
            neg_weight = neg.sum().item() + 1e-12
            loss = (self.weights[0] * pos * loss / pos_weight + self.weights[1] * neg * loss / neg_weight).sum()

            return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5], n_labels=1):  # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights
        self.n_labels = n_labels

    def forward(self, logit, truth, smooth=1e-5):
        if (self.n_labels == 1):
            batch_size = len(logit)
            logit = logit.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            assert (logit.shape == truth.shape)
            p = logit.view(batch_size, -1)
            t = truth.view(batch_size, -1)
            w = truth.detach()
            w = w * (self.weights[1] - self.weights[0]) + self.weights[0]
            # p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
            # t = w*(t*2-1)
            p = w * (p)
            t = w * (t)
            intersection = (p * t).sum(-1)
            union = (p * p).sum(-1) + (t * t).sum(-1)
            dice = 1 - (2 * intersection + smooth) / (union + smooth)
            # print "------",dice.data

            loss = dice.mean()
            return loss


class WeightedDiceBCE(nn.Module):
    def __init__(self,dice_weight=1,BCE_weight=1,n_labels=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5],n_labels=n_labels)
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5],n_labels=n_labels)
        self.n_labels = n_labels
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        # print("2",np.sum(tmp))
        targets[targets>0] = 1
        targets[targets<=0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        # print "dice",dice
        # print "focal",focal
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, pred, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # pred = F.sigmoid(pred)
        # flatten label and prediction tensors
        pred = pred.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (pred * targets).sum()
        total = (pred + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU

"""BCE + IoU Loss"""
class BceIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceIoULoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.iou = IoULoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        iouloss = self.iou(pred, target)

        loss = iouloss + bceloss

        return loss

""" Structure Loss: https://github.com/DengPingFan/PraNet/blob/master/MyTrain.py """
class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()

