# PLHN BreastDM Adaptation

This folder adapts the official PLHN code to the current BreastSeg fixed train/val/test split.

The original PLHN implementation expects 3D NIfTI cases:

```text
data_folder/train1.txt
train/<patient>/P0.nii.gz
train/<patient>/P1.nii.gz
train/<patient>/GT.nii.gz
```

The current BreastSeg project keeps raw BreastDM slices under:

```text
../seg/{train,val,test}/images/<patient>/<phase>/<slice>.jpg
../seg/{train,val,test}/labels/<patient>/<phase>/<slice>.png
```

This adapter converts those slices into PLHN-style 3D cases, then trains PLHN with patch sampling and sliding-window validation/test.

## Prepare Data

The training script prepares data automatically when `convert_from_seg: true`.
To run conversion only:

```bash
python reproduce_plhn/prepare_breastdm_3d.py --seg_root ../seg --output_root ./reproduce_plhn/BreastDM_PLHN_3D
```

Default phase mapping:

- `P0.nii.gz`: `VIBRANT`
- `P1.nii.gz`: `VIBRANT+C8`
- `GT.nii.gz`: labels from `VIBRANT`

Only slices that have pre, post, and label files are converted. This mirrors
the existing fixed-split reproduction scripts, which train/evaluate on labeled
slice pairs from `seg`.

Change `pre_phase`, `post_phase`, or `label_phase` in the YAML if needed.

## Train

Run from `PLHN-main`:

```bash
python reproduce_plhn/train_fixed_split.py --config reproduce_plhn/configs/plhn_breastdm_3d.yaml
```

For a short smoke run:

```bash
python reproduce_plhn/train_fixed_split.py --config reproduce_plhn/configs/plhn_breastdm_3d.yaml --epochs 1
```

## Outputs

Default outputs stay inside `PLHN-main/reproduce_plhn`:

```text
BreastDM_PLHN_3D/
results_plhn_breastdm_3d/
  best_model.pth
  last_model.pth
  training_log.csv
  metrics_test.json
  predicted_masks/
```

Metrics include volume-level mean and global voxel Dice, IoU, recall, precision, and accuracy.
