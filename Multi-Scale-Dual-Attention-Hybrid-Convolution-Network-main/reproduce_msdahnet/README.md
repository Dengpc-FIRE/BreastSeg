# MSDAHNet BreastDM Reproduction

This reproduction is based on the official released implementation in the parent directory. The network structure is not redesigned. The only compatibility edits made to `resunet.py` are:

- `DualA_Net` now inherits `nn.Module`.
- `in_channels` and `num_classes` are configurable.
- import-time model printing is guarded by `if __name__ == "__main__"`.

Unused modules in the official file are preserved because they may document the released implementation, even when they are not called by `forward`.

## Paper Target

Reported BreaDM / BreastDM metrics:

| Metric | Paper |
| --- | ---: |
| Dice | 0.832 |
| IoU | 0.745 |
| Recall | 0.775 |
| Precision | 0.946 |
| Accuracy | 0.999 |
| HD | 1.715 |

Primary reproduction result is slice-level mean over 5 folds. The scripts also save global pixel-level and patient-level summaries where patient ids can be inferred.

## Data

Default paper-faithful mode uses pre-contrast 2D grayscale images:

```text
reproduce_msdahnet/BreastDM/
  pre_contrast_images/
  masks/
```

If you want to build this local format from this repository's `processed_17ch_dce`, set:

```yaml
data:
  convert_from_processed_17ch: true
```

The converter extracts channel 0 as pre-contrast input and writes images/masks under `reproduce_msdahnet/BreastDM`, keeping new data inside the reproduction folder.

## Input Channel Schemes

Scheme A is default and matches the paper table:

```yaml
data:
  in_channels: 1
  gray_to_rgb: false
model:
  in_channels: 1
```

Scheme B is supported for official-code compatibility:

```yaml
data:
  gray_to_rgb: true
model:
  in_channels: 3
```

When Scheme B is used, logs note: "Because the official code expects 3 input channels, grayscale BreaDM images are repeated into 3 channels."

## Run

From `Multi-Scale-Dual-Attention-Hybrid-Convolution-Network-main`:

```bash
python reproduce_msdahnet/train_5fold.py --config reproduce_msdahnet/configs/msdahnet_breastdm_5fold.yaml
```

Smoke test one fold for one epoch:

```bash
python reproduce_msdahnet/train_5fold.py --fold 0 --epochs 1
```

Evaluate an existing fold checkpoint:

```bash
python reproduce_msdahnet/evaluate.py --fold 0
```

## Outputs

```text
reproduce_msdahnet/results_msdahnet_breastdm_5fold/
  splits/fold_0_train.txt
  splits/fold_0_val.txt
  fold_0/best_model_fold0.pth
  fold_0/last_model_fold0.pth
  fold_0/training_log_fold0.csv
  fold_0/fold_0_summary.json
  summary_5fold.json
  summary_5fold.csv
  reproduction_notes.txt
```

`summary_5fold.json` contains each fold result, mean ± std, paper values, and reproduction-minus-paper differences.
