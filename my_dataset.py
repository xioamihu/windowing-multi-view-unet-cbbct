import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2
import random


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None, view='default'):
        super(DriveDataset, self).__init__()
        data_root = root
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, 'default')) if i.endswith(".png")]
        self.img_list = [os.path.join(data_root, view, i) for i in img_names]
        self.roi_mask = [os.path.join(data_root, "mask", f"mask{i[4:]}") for i in img_names]

    def __getitem__(self, idx):
        
        img = Image.open(self.img_list[idx]).convert('RGB')
        
        mask = Image.open(self.roi_mask[idx]).convert('L')
        mask = np.array(mask) / 255
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # max_size = tuple(max(s) for s in zip(*[[3, 864, 864] if img.shape[0]==3 else [864, 864] for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    # print(batched_imgs.shape)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

