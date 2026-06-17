import torch
import gc
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from tqdm import tqdm
import time

from Model.HCRT import HCRT

from torch.utils.data import DataLoader
from utils import logger, util
from utils.metrics import seg_metric
import torch.nn as nn
import os
from dataset.dataset_lits_testx import Test_all_Datasets, Recompone_tool
from collections import OrderedDict
from monai.utils import set_determinism

set_determinism(seed=3407)

def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image

def test_all(model_name='latest_model.pth'):
    opt = Options_x().parse()
    device = torch.device('cuda:' + opt.gpu_ids if torch.cuda.is_available() else "cpu")

    model = HCRT(inch=2, base_channeel=32, imgsize=[48, 128, 128]).to(device)

    ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + model_name, map_location=device)
    model.load_state_dict(ckpt['model'])
    save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'Pred')

    util.mkdir(save_result_path)
    model.eval()
    log_test = logger.Test_Logger(save_result_path, "results")
    cut_param = {'patch_s': opt.patch_size[0], 'patch_h': opt.patch_size[1], 'patch_w': opt.patch_size[2],
                 'stride_s': opt.patch_stride[0], 'stride_h': opt.patch_stride[1], 'stride_w': opt.patch_stride[2]}
    datasets = Test_all_Datasets(opt.datapath, cut_param, opt.data_folder)
    dice_all = 0.0
    for img_dataset, original_shape, new_shape, mask, file_idx in datasets:
        save_tool = Recompone_tool(original_shape, new_shape, cut_param)
        dataloader = DataLoader(img_dataset, batch_size=opt.test_batch, num_workers=opt.num_threads, shuffle=False)
        with torch.no_grad():
            total_batches = 0
            for pre, pos, sub, gt in tqdm(dataloader):
                pre, pos, sub, gt = pre.to(device), pos.to(device), sub.to(device), gt.to(device)
                pos = pos.unsqueeze(1).type(torch.float32)
                sub = sub.unsqueeze(1).type(torch.float32)
                output = model(torch.cat([pos, sub], 1))
                output = (output >= 0.5).type(torch.float32)
                save_tool.add_result(output.detach().cpu())
                total_batches += 1
        pred = save_tool.recompone_overlap()
        recon = (pred.numpy() > 0.5).astype(np.uint16) * mask
        gt_path = os.path.join(opt.datapath, 'label', '{}_GT.nii.gz'.format(file_idx))
        gt_img = sitk.ReadImage(gt_path)
        gt = sitk.GetArrayFromImage(gt_img)
        DSC, PPV, SEN, ASD = seg_metric(recon, gt)
        dice_all += DSC
        index_results = OrderedDict({'DSC': DSC, 'PPV': PPV, 'SEN': SEN, 'ASD': ASD})
        log_test.update(file_idx, index_results)
        Pred = sitk.GetImageFromArray(np.array(recon))
        Pred.CopyInformation(gt_img)
        result_save_path = os.path.join(save_result_path, file_idx)
        util.mkdir(result_save_path)
        sitk.WriteImage(Pred, os.path.join(result_save_path, 'pred.nii.gz'))
        del pred, recon, Pred, save_tool, gt
        gc.collect()
    print(f"avg dice：{dice_all / 100}")


if __name__ == '__main__':
    test_all('latest_model.pth')
