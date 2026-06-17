import torch
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from dataset.dataset_lits_trainx import Lits_DataSet
from test import test_all

from Model.HCRT import HCRT
from torch.utils.data import DataLoader
from utils.common import adjust_learning_rate
from utils import logger, util
import torch.nn as nn
from utils.metrics import LossAverage, DiceLoss, seg_metric, Dice_Loss

import os
from collections import OrderedDict
import torch.nn.functional as F
import time

class rca_model(nn.Module):
    def __init__(self, base_channeel=2):
        super().__init__()
        self.feat_dim = base_channeel
        self.queue_len = 10

        self.alpha = 8
        self.temperature = 0.07

        self.num_prototypes = 10
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.feat_dim, 1, 1, 1))

        # -----------------------set memory-----------------------------
        for i in range(0, 2):
            self.register_buffer("queue" + str(i), torch.randn(self.queue_len, self.feat_dim))
            self.register_buffer("queue_ptr" + str(i), torch.zeros(1, dtype=torch.long))
        self.momentum = 0.99

    def forward(self, x, label):
        x2 = x
        device = torch.device('cuda:' + "0" if torch.cuda.is_available() else "cpu")
        similarity = F.conv3d(x2, self.prototypes)
        similarity = similarity.permute(0, 2, 3, 4, 1)  # (batch_size, D, H, W, num_prototypes)
        protologist = F.softmax(similarity / self.temperature, dim=-1)
        proto_indices = torch.argmax(protologist, dim=-1, keepdim=True)  # (batch_size, D, H, W, 1)
        proto = torch.zeros_like(x2)
        for i in range(self.num_prototypes):
            mask = (proto_indices == i).squeeze(-1).unsqueeze(1)
            proto += mask * self.prototypes[i]

        feat_memory = getattr(self, "queue0")
        batch_num = x2.size()[0]
        loss_cl = torch.zeros(1).to(device)
        for i in range(0, batch_num):
            mask_i = protologist[i][0] > (torch.mean(protologist[i][0]))
            mask_lab = label[i][0].float().sum()

            if mask_lab.float().sum() < 50:
                mid_sel = proto[i] * mask_i.float().detach()
                x_mid_pool = mid_sel.reshape(mid_sel.shape[0], -1).sum(1) / mask_i.float().sum().detach()
            else:
                mid_sel = proto[i] * mask_lab.float().detach()
                x_mid_pool = mid_sel.reshape(mid_sel.shape[0], -1).sum(1) / mask_lab.float().sum().detach()

            x_mid_pool_norm = F.normalize(x_mid_pool.unsqueeze(0)).squeeze()
            feat_neg = F.normalize(feat_memory)

            similarity_neg = torch.einsum('c,nc->n', [x_mid_pool_norm, feat_neg.detach()])
            logit_neg = torch.div(similarity_neg, self.temperature)
            max_log = torch.max(logit_neg)
            exp_logit_neg = torch.exp(logit_neg - max_log.detach())
            label_q = torch.arange(1)
            label_q = label_q.expand(self.queue_len, 1).permute(1, 0).reshape(self.queue_len * 1).to(device)

            feat_pos = F.normalize(getattr(self, "queue" + str(1)))
            mask = label_q != 1
            similarity_pos = torch.einsum('c,nc->n', [x_mid_pool_norm, feat_pos.detach()])
            logit_pos = torch.div(similarity_pos, self.temperature)
            logit_pos = logit_pos - max_log.detach()
            exp_logit_pos = torch.exp(logit_pos)

            l_neg = (exp_logit_neg * mask.float().detach()).sum().expand(self.queue_len)
            loss_i_1 = (-(logit_pos - torch.log((l_neg + exp_logit_pos).clamp(min=1e-4)))).mean()
            loss_cl += loss_i_1

        for j in range(0, batch_num):
            self._dequeue_and_enqueue_v2(proto[j], label[j])

        return loss_cl / batch_num


    @torch.no_grad()
    def _dequeue_and_enqueue_v2(self, x, label):
        orgf = x
        mask2 = label[0]

        if mask2.float().sum() > 10:
            x = x * mask2.float()
            embedding = x.reshape(x.shape[0], -1).sum(1) / ((mask2.float()).sum() + 1)

            queue_i = getattr(self, "queue" + str(1))
            queue_ptr_i = getattr(self, "queue_ptr" + str(1))
            ptr = int(queue_ptr_i)
            queue_i[ptr:ptr + 1] = queue_i[ptr:ptr + 1] * self.momentum + embedding * (
                    1 - self.momentum)
            ptr = (ptr + 1) % self.queue_len
            queue_ptr_i[0] = ptr

        mask2 = 1 - label[0]
        if mask2.float().sum() > 10:
            x = orgf * mask2.float()
            embedding = x.reshape(x.shape[0], -1).sum(1) / ((mask2.float()).sum() + 1)

            queue_i = getattr(self, "queue" + str(0))
            queue_ptr_i = getattr(self, "queue_ptr" + str(0))
            ptr = int(queue_ptr_i)
            queue_i[ptr:ptr + 1] = queue_i[ptr:ptr + 1] * self.momentum + embedding * (1 - self.momentum)
            ptr = (ptr + 1) % self.queue_len
            queue_ptr_i[0] = ptr


def train(train_dataloader, epoch):
    print("=======Epoch:{}======Learning_rate:{}=========".format(epoch, optimizer.param_groups[0]['lr']))

    Loss = LossAverage()
    DICE_Loss = LossAverage()
    BCE_Loss = LossAverage()
    CL_Loss = LossAverage()

    model.train()
    rca = rca_model().to(device)
    for i, (pre, pos, sub, gt) in enumerate(train_dataloader):  # inner loop within one epoch
        b, c, l, w, e = pre.shape[0], pre.shape[1], pre.shape[2], pre.shape[3], pre.shape[4],
        pre = pre.view(-1, 1, l, w, e).to(device)
        pos = pos.view(-1, 1, l, w, e).to(device)
        sub = sub.view(-1, 1, l, w, e).to(device)
        gt = gt.view(-1, 1, l, w, e).to(device)
        pred = model(torch.cat([pos, sub], 1))  # pos sub
        pred_soft = torch.softmax(pred, dim=1)
        outpd_soft = torch.softmax(gt, dim=1)
        loss_cl = rca(pred_soft, outpd_soft)
        Dice_loss = dice_loss(pred, gt)
        Bce_loss = bce_loss(pred, gt.squeeze(1).long())

        loss = Bce_loss + 5 * Dice_loss + 0.2 * loss_cl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer, epoch, opt)

        Loss.update(loss.item(), pre.size(0))
        DICE_Loss.update(Dice_loss.item(), pre.size(0))
        BCE_Loss.update(Bce_loss.item(), pre.size(0))
        CL_Loss.update(loss_cl.item(), pre.size(0))

    return OrderedDict({'Loss': Loss.avg, 'DICE_Loss': DICE_Loss.avg, 'BCE_Loss': BCE_Loss.avg, 'CL_Loss': CL_Loss.avg})



if __name__ == '__main__':
    opt = Options_x().parse()
    device = torch.device('cuda:' + opt.gpu_ids if torch.cuda.is_available() else "cpu")


    model = HCRT(inch=2, base_channeel=32, imgsize=[48, 128, 128]).to(device)
    ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + 'latest_model.pth', map_location=device)
    model.load_state_dict(ckpt['model'])
    save_path = opt.checkpoints_dir

    dice_loss = Dice_Loss(2)
    bce_loss = torch.nn.CrossEntropyLoss()

    save_result_path = os.path.join(save_path, opt.task_name)
    util.mkdir(save_result_path)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                           weight_decay=1e-5)

    model_save_path = os.path.join(save_result_path, 'model')
    util.mkdir(model_save_path)
    logger_save_path = os.path.join(save_result_path, 'logger')
    util.mkdir(logger_save_path)
    log_train = logger.Train_Logger(logger_save_path, "train_log")

    train_dataset = Lits_DataSet(opt.datapath, 'partial', opt.patch_size, opt.data_folder)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, \
                                  num_workers=opt.num_threads, shuffle=True)

    for epoch in range(opt.epoch):
        epoch = epoch + 1

        t1=time.time()
        train_log= train(train_dataloader,epoch)
        print(f"Time Elapsed: {time.time() - t1:.2f} seconds")

        log_train.update(epoch, train_log)

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, os.path.join(model_save_path, 'latest_model.pth'))

        if epoch % opt.model_save_fre == 0:
            torch.save(state, os.path.join(model_save_path, 'model_' + np.str_(epoch) + '.pth'))

        torch.cuda.empty_cache()

    test_all('latest_model.pth')
