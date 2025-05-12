# [A windowing-based multi-view u-net for tumor segmentation in cone-beam breast CT](https://www.sciencedirect.com/science/article/pii/S1746809424015404)

## Overview
![framework](README.assets/framework.jpg)

## Environment
- Anaconda3
- python3.7/3.8
- pytorch 1.8.0

## Data Structure
```
├── data
│   ├── test
│   │   └── Stores the test dataset, which is used to evaluate the performance of the trained model.
│   ├── train
│   │   ├── default
│   │   │   └── The sub-directory for storing training data, which may contain basic training samples (defalut view).
│   │   ├── mask
│   │   │   └── Stores the mask files corresponding to the training data, which are used for annotation in segmentation.
│   │   ├── multi-view1
│   │   │   └── One type of multi-view training data, which are used for multi-view related model training.
│   │   ├── multi-view2
│   │   │   └── Second type of multi-view training data.
│   │   └── multi-view3
│   │       └── Third type of multi-view training data.
│   └── val
│       └── Stores the validation dataset.
├── data_dicom
│   └── Stores dicom series.
├── roi
    └── Stores mask files(.nii).
```
## Data Precessing
Use windowing processing to convert DICOM series and NII mask files into images.
```
python data_slice.py
```
## Train
```
python train_resnet152.py
```
where you can change the type of training model and multi-view training data.

## Test
```
python predict.py
```
For an input image, you can get tumor predicion in `./out.png`. If you have ground truth, metrics including IoU, Dice, sensitivity, specificity, and HD95 will be computed.

## Citation
```
@article{peng2025windowing,
  title={A windowing-based multi-view u-net for tumor segmentation in cone-beam breast CT},
  author={Peng, Die and Su, Tong and Zheng, Yineng and Ouyang, Zubin and Chen, Bohui and Chen, Binlin and He, Jiaqi and Ma, Qiang and Yang, Liang and Li, Haofeng and others},
  journal={Biomedical Signal Processing and Control},
  volume={103},
  pages={107482},
  year={2025},
  publisher={Elsevier}
}
```
