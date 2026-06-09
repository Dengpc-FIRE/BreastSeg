# PDF-UNet BreastDM Adaptation

This folder adapts the official PDF-UNet implementation to BreastDM / BreaDM DCE-MRI as a baseline.

Important distinction:

- Original PDF-UNet setting: breast ultrasound datasets.
- Our adaptation setting: BreastDM DCE-MRI.

The official pyramid-dilated U-shaped network is preserved. The source model in `../PDF-UNet.py` was only made configurable for `in_channels`, `num_classes`, feature widths, dropout, bilinear upsampling, and residual projection alignment.

## Data

Default configs read:

```text
processed_breastdm/
  train/data
  train/GT
  val/data
  val/GT
  test/data
  test/GT
```

Prepare this from an existing `processed_17ch_dce` folder:

```bash
python reproduce_pdfunet/prepare_breastdm.py --source ../processed_17ch_dce --output ./processed_breastdm --input-mode single_channel_pre
```

For adapted multi-phase DCE input, keep `.npy` files and set the config to `input_mode: "multi_phase"` and `in_channels: 17`.

## Train

Focal Tversky, matching the strongest original loss setting:

```bash
python reproduce_pdfunet/train.py --config reproduce_pdfunet/configs/pdfunet_breastdm_focaltversky.yaml
```

Dice + BCE comparison:

```bash
python reproduce_pdfunet/train.py --config reproduce_pdfunet/configs/pdfunet_breastdm_dicebce.yaml
```

Compare completed fixed-split runs:

```bash
python reproduce_pdfunet/compare_losses.py --root results_pdfunet_breastdm
```

## Outputs

Each run saves:

```text
best_model.pth
last_model.pth
training_log.csv
config.yaml
metrics_test.json
predicted_masks/
```

Metrics include slice-level mean, patient-level mean when patient ids can be inferred, and global pixel-level Dice/IoU/Recall/Precision/Accuracy. HD95 is reported for slice/patient-level masks; global HD95 is `nan` because global masks are flattened.

