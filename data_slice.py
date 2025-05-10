import SimpleITK as sitk
import cv2
import matplotlib
matplotlib.use('TkAgg')
import nibabel as nib
import numpy as np
import os
from shutil import copy, rmtree
import pydicom
from PIL import Image
import pandas as pd

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # rmtree(file_path)
        # os.makedirs(file_path)
        return file_path
    os.makedirs(file_path)
    return file_path

save_crop = True
img_path = "data_dicom" # dicom root
mask_path = 'roi' # mask(nii root)

root = "data/train"
default_view = mk_file(os.path.join(root, "default"))
multi_view1 = mk_file(os.path.join(root, "multi-view1"))
multi_view2 = mk_file(os.path.join(root, "multi-view2"))
multi_view3 = mk_file(os.path.join(root, "multi-view3"))
mask_img = mk_file(os.path.join(root, "mask"))

def normalizatingImage(img, fc, fw):
    fmin = (2.0 * fc - fw) / 2 + 0.5
    fmax = (2.0 * fc + fw) / 2 + 0.5
    img = np.clip(img, fmin, fmax)
    img = (img - fmin) / (fmax - fmin) * 255
    img = Image.fromarray(img)
    if img.mode == "F":
        img = img.convert('RGB')
    return img

# read dicom series
def readImage(imgPath):
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(imgPath) 
    reader.SetFileNames(img_names)
    image = reader.Execute()
    img = sitk.GetArrayFromImage(image)
    return img, img_names

img_class = [cla for cla in os.listdir(img_path)
             if os.path.isdir(os.path.join(img_path, cla))]
mask_class = [cla for cla in os.listdir(mask_path)
              if cla.endswith("nii")]

for index, cla in enumerate(img_class):

    # matching Data(dicom), mask(nii), example:"xxc1" , "xxc2.nii"
    mask_name = cla[:-1] + "2.nii"

    if mask_name in mask_class:
        img_in = os.path.join(img_path, cla)
        mask_in = os.path.join(mask_path, mask_name)
        img_original, img_names = readImage(img_in)

        ds = pydicom.read_file(img_names[0])
        fc = ds.WindowCenter
        fw = ds.WindowWidth

        mask = nib.load(mask_in).get_fdata()
        mask = np.transpose(mask)

        if mask.shape == img_original.shape:
            # slice in the z direction
            for i in range(mask.shape[0]):
                if (mask[i, :, :] == 0).all():
                    continue
                else:
                    mask_slice = mask[i, :, :] * 255
                    mask_slice = Image.fromarray(mask_slice).convert('L')

                    ds = pydicom.read_file(img_names[i])
                    img_slice = ds.pixel_array

                    img_slice0 = normalizatingImage(img_slice, fc, fw).convert('L')
                    width, height = img_slice0.size
                    
                    img_slice2 = normalizatingImage(img_slice, 28, 277).convert('L')
                    img_slice3 = normalizatingImage(img_slice, 109, 527).convert('L')
                    img_slice4 = normalizatingImage(img_slice, -86, 598).convert('L')

                    result_image1 = Image.merge('RGB', [img_slice0, img_slice2, img_slice3])
                    result_image2 = Image.merge('RGB', [img_slice0, img_slice2, img_slice4])
                    result_image3 = Image.merge('RGB', [img_slice0, img_slice3, img_slice4])
                    
                    # img_slice0.save(f"{os.path.join(default_view, f'norm_{img_class[index]}_{i}.png')}")
                    # mask_slice.save(f"{os.path.join(mask_img, f'mask_{img_class[index]}_{i}.png')}")
                    # result_image1.save(f"{os.path.join(multi_view1, f'norm_{img_class[index]}_{i}.png')}")
                    # result_image2.save(f"{os.path.join(multi_view2, f'norm_{img_class[index]}_{i}.png')}")
                    # result_image3.save(f"{os.path.join(multi_view3, f'norm_{img_class[index]}_{i}.png')}")
                    
                    if save_crop:
                        img_slice0 = np.array(img_slice0)
                        mask_slice = np.array(mask_slice)
                        result_image1 = np.array(result_image1)
                        result_image2 = np.array(result_image2)
                        result_image3 = np.array(result_image3)
                        
                        pos = np.where(img_slice0 > 10)
                        pos_x = 16 - (np.max(pos[0]) - np.min(pos[0])) % 16
                        pos_y = 16 - (np.max(pos[1]) - np.min(pos[1])) % 16
                        
                        if np.min(pos[0])-pos_x//2 < 0 or np.min(pos[1]) - pos_y // 2 < 0:
                            img_slice0 = img_slice0[np.min(pos[0]):np.max(pos[0]) + pos_x,
                                                    np.min(pos[1]):np.max(pos[1]) + pos_y]
                            result_image1 = result_image1[np.min(pos[0]):np.max(pos[0]) + pos_x,
                                                          np.min(pos[1]):np.max(pos[1]) + pos_y, :]
                            result_image2 = result_image2[np.min(pos[0]):np.max(pos[0]) + pos_x,
                                                          np.min(pos[1]):np.max(pos[1]) + pos_y, :]
                            result_image3 = result_image3[np.min(pos[0]):np.max(pos[0]) + pos_x,
                                                          np.min(pos[1]):np.max(pos[1]) + pos_y, :]
                            mask_slice = mask_slice[np.min(pos[0]):np.max(pos[0]) + pos_x,
                                                    np.min(pos[1]):np.max(pos[1]) + pos_y]
                        else:
                            img_slice0 = img_slice0[np.min(pos[0]) - pos_x // 2:np.max(pos[0]) + pos_x - pos_x // 2,
                                                    np.min(pos[1]) - pos_y // 2:np.max(pos[1]) + pos_y - pos_y // 2]
                            result_image1 = result_image1[np.min(pos[0]) - pos_x // 2:np.max(pos[0]) + pos_x - pos_x // 2,
                                                          np.min(pos[1]) - pos_y // 2:np.max(pos[1]) + pos_y - pos_y // 2, :]
                            result_image2 = result_image2[np.min(pos[0]) - pos_x // 2:np.max(pos[0]) + pos_x - pos_x // 2,
                                                          np.min(pos[1]) - pos_y // 2:np.max(pos[1]) + pos_y - pos_y // 2, :]
                            result_image3 = result_image3[np.min(pos[0]) - pos_x // 2:np.max(pos[0]) + pos_x - pos_x // 2, 
                                                          np.min(pos[1]) - pos_y // 2:np.max(pos[1]) + pos_y - pos_y // 2, :]
                            mask_slice = mask_slice[np.min(pos[0]) - pos_x // 2:np.max(pos[0]) + pos_x - pos_x // 2,
                                                    np.min(pos[1]) - pos_y // 2:np.max(pos[1]) + pos_y - pos_y // 2]
                        
                        img_slice0 = Image.fromarray(img_slice0)
                        result_image1 = Image.fromarray(result_image1)
                        result_image2 = Image.fromarray(result_image2)
                        result_image3 = Image.fromarray(result_image3)
                        mask_slice = Image.fromarray(mask_slice)

                    img_slice0.save(f"{os.path.join(default_view, f'norm_{img_class[index]}_{i}.png')}")
                    mask_slice.save(f"{os.path.join(mask_img, f'mask_{img_class[index]}_{i}.png')}")
                    result_image1.save(f"{os.path.join(multi_view1, f'norm_{img_class[index]}_{i}.png')}")
                    result_image2.save(f"{os.path.join(multi_view2, f'norm_{img_class[index]}_{i}.png')}")
                    result_image3.save(f"{os.path.join(multi_view3, f'norm_{img_class[index]}_{i}.png')}")
        
        else:
            print(f'{cla} data size: {img_original.shape} and mask size: {mask.shape} not matching！')

    else:
        print(f'{cla} has no mask files！')
