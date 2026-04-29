import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

# 1. 确保 model1.py 在当前目录下，以便加载 HybridTemporalNet
from model1 import HybridTemporalNet

# ==========================================
# 配置参数 (请根据你的实际路径修改)
# ==========================================
TEST_DATA_DIR = './cropped_dataset/test' # 之前离线裁剪好的测试集路径
MODEL_WEIGHTS = './best_seg_cropped.pth'  # 你训练完成的最佳权重
OUTPUT_VIS_DIR = './final_evaluation_results' # 结果保存路径
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8

# ==========================================
# 数据集类
# ==========================================
class Stage2TestDataset(Dataset):
    def __init__(self, root_dir):
        self.data_dir = os.path.join(root_dir, 'data')
        self.gt_dir = os.path.join(root_dir, 'GT')
        self.ids = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]

    def __len__(self): 
        return len(self.ids)

    def __getitem__(self, i):
        fname = self.ids[i]
        # 加载 9 通道数据并归一化
        img = (np.load(os.path.join(self.data_dir, fname)).transpose(2,0,1).astype(np.float32) / 255.0)
        # 加载 Mask
        mask = cv2.imread(os.path.join(self.gt_dir, fname.replace('.npy', '.png')), 0)
        mask = (mask > 127).astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(mask).unsqueeze(0), fname

# ==========================================
# 指标计算函数
# ==========================================
def compute_metrics(pred_logits, target):
    # 概率二值化
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    
    # 使用 smp 库的标准指标
    tp, fp, fn, tn = smp.metrics.get_stats(pred.long(), target.long(), mode='binary')
    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
    
    return dice.item(), iou.item()

# ==========================================
# 主评估程序
# ==========================================
def run_evaluation():
    print(f"正在启动评估程序... 使用设备: {DEVICE}")
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    # 1. 初始化模型并加载权重
    model = HybridTemporalNet(base_ch=32, depth=4).to(DEVICE)
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"错误: 找不到权重文件 {MODEL_WEIGHTS}")
        return

    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    print(f"成功加载最佳模型权重: {MODEL_WEIGHTS}")

    # 2. 准备数据
    dataset = Stage2TestDataset(TEST_DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"测试集样本总数: {len(dataset)}")

    # 3. 开始推理
    test_dices = []
    test_ious = []
    
    with torch.no_grad():
        for i, (imgs, masks, fnames) in enumerate(tqdm(loader, desc="正在计算测试指标")):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            # 混合精度推理 (可选)
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
            
            # 逐个计算指标
            # 注意: 为了指标精确，我们对 batch 内每个样本分别评估
            for j in range(imgs.size(0)):
                d, iou = compute_metrics(preds[j:j+1], masks[j:j+1])
                test_dices.append(d)
                test_ious.append(iou)

                # 保存前 20 张对比图用于论文展示
                if i * BATCH_SIZE + j < 20:
                    p_np = (torch.sigmoid(preds[j, 0]) > 0.5).cpu().numpy().astype(np.uint8) * 255
                    m_np = (masks[j, 0].cpu().numpy() * 255).astype(np.uint8)
                    i_np = (imgs[j, 0].cpu().numpy() * 255).astype(np.uint8) # 展示V0通道
                    
                    # 拼接: 原图 | 真值 | 预测
                    combined = np.hstack([i_np, m_np, p_np])
                    cv2.imwrite(os.path.join(OUTPUT_VIS_DIR, f"result_{fnames[j].split('.')[0]}.png"), combined)

    # 4. 输出最终统计结果
    final_dice = np.mean(test_dices)
    final_iou = np.mean(test_ious)
    
    print("\n" + "="*40)
    print(">>>> 最终测试统计结果 <<<<")
    print(f"Mean Test Dice: {final_dice:.4f}")
    print(f"Mean Test IoU:  {final_iou:.4f}")
    print(f"测试完成! 对比图已保存至: {OUTPUT_VIS_DIR}")
    print("="*40)

    # 将结果写入文本文件，方便复制到论文
    with open(os.path.join(OUTPUT_VIS_DIR, "test_metrics.txt"), "w") as f:
        f.write(f"Mean Test Dice: {final_dice:.4f}\n")
        f.write(f"Mean Test IoU:  {final_iou:.4f}\n")

if __name__ == "__main__":
    run_evaluation()