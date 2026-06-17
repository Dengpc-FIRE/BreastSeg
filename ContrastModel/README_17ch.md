# 17-Channel BreastDM Contrast Training

This folder contains unified 17-channel adapters for the 12 contrast models.
All changes are kept under `ContrastModel/`.

## Data

2D models read the main processed slice format:

```text
processed_17ch_dce/{train,val,test}/data/*.npy  # [H, W, 17] or [17, H, W]
processed_17ch_dce/{train,val,test}/GT/*.png
```

3D models read the raw patient format and cache built volumes under
`ContrastModel/dataset/processed_3d_17ch/`:

```text
seg/{train,val,test}/images/<patient>/VIBRANT, VIBRANT+C1..C8, SUB1..SUB8
seg/{train,val,test}/labels/<patient>/<label_phase>
```

If your raw root is not `seg`, change `data.raw_dataset_root` in each YAML.

## Per-Model Entry Points

Each model has:

```text
configs/breastdm_17ch.yaml
train_17ch.py
test_17ch.py
```

Example:

```bash
python ContrastModel/Pytorch-UNet-master/train_17ch.py
python ContrastModel/Pytorch-UNet-master/test_17ch.py
```

Common CLI overrides:

```bash
python <model>/train_17ch.py --epochs 1 --lr 0.0001 --batch-size 2 --num-workers 0
python <model>/test_17ch.py --checkpoint <model>/checkpoints/best_model.pth
```

Outputs are written inside each model directory:

```text
checkpoints/best_model.pth
checkpoints/latest_model.pth
training_log.csv
test_results/summary.txt
test_results/metrics.csv
```

The test summary prints and saves:

```text
mean_dice
mean_iou
mean_hd95
mean_sensitivity
mean_precision
mean_accuracy
```

## Model Modes

2D: TransUNet, Mobile-U-ViT, EMCAD, DeepLabV3Plus, Pytorch-UNet, MSDAHNet,
Attention-Gated-Networks, PDPNet, UNetPlusPlus.

PDPNet defaults to its `DPKNet` segmentation branch on full 17-channel slices
(`use_location_branch: false`) because the original location-guided crop path is
numerically fragile on this dataset. Set it to `true` only if you specifically
want to reproduce the original crop-guided behavior.

3D: HCRT, PLHN, nnU-Net.

## nnU-Net

The nnU-Net adapter writes data to:

```text
ContrastModel/nnUNet-master/raw/Dataset501_BreastDM17
ContrastModel/nnUNet-master/preprocessed
ContrastModel/nnUNet-master/results
```

Run:

```bash
python ContrastModel/nnUNet-master/prepare_17ch_nnunet.py
python ContrastModel/nnUNet-master/train_17ch.py
python ContrastModel/nnUNet-master/test_17ch.py
```

The custom trainer `nnUNetTrainerBreastDM17` reads `epochs`, `lr`, and
`weight_decay` from `configs/breastdm_17ch.yaml`.
