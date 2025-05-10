import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from torch.utils.data import random_split
from medpy.metric.binary import hd, hd95

# from src import UNet, VGG16UNet, DensenetUnet, InceptionUnet, EfficientnetUnet, Resnet50Unet, Resnet152Unet
from src import UNet, VGG16UNet, MobileV3Unet, DensenetUnet, InceptionUnet, EfficientnetUnet,Resnext50Unet, Resnet18Unet, Resnet34Unet, Resnet50Unet, Resnet101Unet, Resnet152Unet, Resnext50Unet, Resnext101Unet

import numpy as np
from collections import OrderedDict
from typing import Dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet18, resnet34
from src.unet import Up, OutConv
from my_dataset import DriveDataset
import transforms as T


class SegmentationPresetTrain:
    def __init__(self, base_size, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, affine_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = []
        # trans = [T.pad_if_smaller(base_size)]
        # trans.append(T.RandomResize(base_size))
        # if rotate_prob > 0:
        #     trans.append(T.RotateAngle(rotate_prob))
        # if affine_prob > 0:
        #     trans.append(T.Affine(affine_prob))
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            # T.RandomCrop(224),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 224

    if train:
        return SegmentationPresetTrain(base_size, mean=mean, std=std)
    else:
        # return SegmentationPresetEval(mean=mean, std=std)
        return SegmentationPresetEval(base_size, mean=mean, std=std)

def predict_single_image():
    classes = 1  # exclude background
    # img_path = "./data/train/default/norm_1845707c1_206.png"
    img_path = "./data/train/multi-view1/norm_1845707c1_206.png"
    # img_path = "./data/train/multi-view2/norm_1845707c1_206.png"
    # img_path = "./data/train/multi-view3/norm_1845707c1_206.png"
    roi_mask_path = "./data/train/mask/mask_1845707c1_206.png"
    
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # model = UNet(in_channels=3, num_classes=classes+1, base_c=32)
    # weights_path = "./checkpoints/unet.pth"
    # model = VGG16UNet(num_classes=classes+1)
    # weights_path = "./checkpoints/vgg16_unet.pth"
    # model = Resnet18Unet(num_classes=classes+1)
    # weights_path = "./checkpoints/resnet18_unet.pth"
    # model = Resnet34Unet(num_classes=classes+1)
    # weights_path = "./checkpoints/resnet34_unet.pth"
    # model = Resnet50Unet(num_classes=classes+1)
    # weights_path = "./checkpoints/resnet50_unet.pth"
    # model = Resnet101Unet(num_classes=classes+1)
    # weights_path = "./checkpoints/resnet101_unet.pth"
    
    model = Resnet152Unet(num_classes=classes+1)
    weights_path = "./checkpoints/resnet152_unet_mv1.pth"
    # weights_path = "./checkpoints/resnet152_unet.pth"
    # weights_path = "./checkpoints/resnet152_unet_default_mv2.pth"
    # weights_path = "./checkpoints/resnet152_unet_mv2.pth"
    # weights_path = "./checkpoints/resnet152_unet_mv2_mv3.pth"
    # weights_path = "./checkpoints/resnet152_unet_mv3.pth"
    # weights_path = "./checkpoints/resnet152_unet_mv3_aug.pth"
    
    # model = Resnext50Unet(num_classes=classes+1)
    # weights_path = "./checkpoints/resnext50_unet.pth"
    # model = Resnext101Unet(num_classes=classes+1)
    # weights_path = "./checkpoints/resnext101_unet.pth"
    # model = DensenetUnet(num_classes=classes+1)
    # weights_path = "./checkpoints/densenet121_unet.pth"
    # model = InceptionUnet(num_classes=classes+1)
    # weights_path = "./checkpoints/inceptionv3_unet.pth"
    # model = EfficientnetUnet(num_classes=classes+1)
    # weights_path = "./checkpoints/efficientnet_b7_unet.pth"
    
    assert os.path.exists(weights_path), f"weights {weights_path} not found."

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img) / 255
    roi_img[roi_img < 0.1] = 0
    roi_img[roi_img >= 0.1] = 1

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        # t_start = time_synchronized()
        output = model(img.to(device))
        # t_end = time_synchronized()
        # print("inference+NMS time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        IoU = np.sum(prediction * roi_img) / (np.sum(prediction) + np.sum(roi_img) - np.sum(prediction * roi_img))
        dice = 2 * np.sum(prediction * roi_img) / (np.sum(prediction) + np.sum(roi_img))
        print(f"IoU:{IoU}")
        print(f"Dice:{dice}")
        prediction[prediction == 1] = 255
        mask = Image.fromarray(prediction)
        mask.save("out.png")


def predict_images_in_dir():
    classes = 1  

    img_path = "./data/train/default"
    roi_mask_path = "./data/train/mask"
    
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("using {} device.".format(device))
    
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)
    weights_path = "./checkpoints/unet.pth"

    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    IoU_list = []
    dice_list = []
    sen_list = []
    spe_list = []
    IoU_low = []
    cla_low = []
    tumor_s = []
    hd_list = []
    hd95_list = []

    for index, cla in enumerate([i for i in os.listdir(img_path) if i.endswith(".png")]): # os.listdir(img_path)

        predict_name = "predict" + cla[4:]
        mask_name = os.path.join(roi_mask_path, f"mask{cla[4:]}")
        roi_img = Image.open(mask_name).convert('L')
        roi_img = np.array(roi_img) / 255
        roi_img[roi_img < 0.1] = 0
        roi_img[roi_img >= 0.1] = 1

        original_img = Image.open(os.path.join(img_path, cla)).convert('RGB')

        data_transform = transforms.Compose(
                                             # [transforms.CenterCrop(512),
                                             # transforms.RandomCrop(224),
                                             [transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval() 
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)

            output = model(img.to(device))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0

            IoU = np.sum(prediction * roi_img) / (np.sum(prediction) + np.sum(roi_img) - np.sum(prediction * roi_img))
            dice = 2 * np.sum(prediction * roi_img) / (np.sum(prediction) + np.sum(roi_img))
            sen = np.sum(prediction * roi_img) / np.sum(roi_img)
            spe = np.sum((1 - prediction) * (1 - roi_img)) / np.sum(1 - roi_img)
            if np.sum(prediction) > 0:
                hausdorff_distance = hd(prediction, roi_img)
                hausdorff_distance95 = hd95(prediction, roi_img)
                hd_list.append(hausdorff_distance)
                hd95_list.append(hausdorff_distance95)
                
            IoU_list.append(IoU)
            dice_list.append(dice)
            sen_list.append(sen)
            spe_list.append(spe)
            tumor_s.append(np.sum(roi_img))
  
    print(f'mean IoU: {np.mean(IoU_list)}')
    print(f'mean Dice: {np.mean(dice_list)}')
    print(f'mean Sensitivity: {np.mean(sen_list)}')
    print(f'mean specificity: {np.mean(spe_list)}')
    print(f'mean HD: {np.mean(hd_list)}') #, np.std(hd_list)
    print(f'mean HD95: {np.mean(hd95_list)}')


if __name__ == '__main__':
    predict_single_image()
    # predict_images_in_dir()