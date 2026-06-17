# HCRT: Hybrid Network with Correlation-Aware Region Transformer for Breast Tumor Segmentation in DCE-MRI
## Requirements
Some important required packages include:
* Python==3.8
* torch==1.9.0
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......
## Dataset
A total of three datasets were used in our paper, among which two private datasets were breast cancer and thymus tumor, and the other dataset was a publicly available breast tumor dataset. If you wish to download this publicly available dataset, please refer to the relevant [paper](https://www.cell.com/patterns/fulltext/S2666-3899(23)00195-2?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2666389923001952%3Fshowall%3Dtrue)/[ZENODO](https://zenodo.org/records/8068383).

Additionally, we provide a publicly available breast cancer [dataset](https://drive.google.com/file/d/1ciSV337l9uyoou2GfbSRHPxHH9r6uxC9/view?usp=sharing) after processing.
```
YN
│
├── data_folder
│   └── test5.txt
├── breastmask
│   ├── Yunnan_1.nii.gz
│   ├── ...
│   └── Yunnan_100.nii.gz 
├── image_p0
│   ├── Yunnan_1_P0.nii.gz
│   ├── ...
│   └── Yunnan_100_P0.nii.gz
├── image_p1
│   ├── Yunnan_1_P1.nii.gz
│   ├── ...
│   └── Yunnan_100_P1.nii.gz
└── label
    ├── Yunnan_1_GT.nii.gz
    ├── ...
    └── Yunnan_100_GT.nii.gz
```
If you require MRI breast tumor segmentation weights, download the [model checkpoint](https://drive.google.com/drive/folders/1XjBD-ylWbvKE4ND7yGjbaiE2_dM9Mw8l?usp=drive_link).
## Whole Breast Segmentation Model
* The whole breast segmentation process can be used to remove the oversegmentation on non-breast regions.
* Whole breast segmentation model is available at: [https://github.com/ZhangJD-ong/AI-assistant-for-breast-tumor-segmentation](https://github.com/ZhangJD-ong/AI-assistant-for-breast-tumor-segmentation)
## Training
Train the model and infers.
```
python train.py
```
## Evaluation
1.Clone the repo.

2.Download the [dataset](https://drive.google.com/file/d/1ciSV337l9uyoou2GfbSRHPxHH9r6uxC9/view?usp=sharing) to the following path:
```
YN
```
3.Download the [model](https://drive.google.com/drive/folders/1XjBD-ylWbvKE4ND7yGjbaiE2_dM9Mw8l?usp=drive_link) weights and paste *latest_model.pth* in the following path:
```
checkpoints\HCRT
```
4.If you only need to test the model.
```
python test.py
```