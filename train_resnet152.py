import os
import time
import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import random_split

from src import UNet, VGG16UNet, MobileV3Unet, DensenetUnet, InceptionUnet, EfficientnetUnet, Resnet18Unet, Resnet34Unet, Resnet50Unet, Resnet101Unet, Resnet152Unet, Resnext50Unet, Resnext101Unet, Resnet152UnetAttn
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
import transforms as T
import numpy as np


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


def create_model(num_classes):

    # ---------UNet-------------
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)

    # ---------VGG16UNet-------------
    # model = VGG16UNet(num_classes=num_classes)
    # out_channel = model.backbone['0'].out_channels
    # model.backbone['0'] = nn.Conv2d(1, out_channel, kernel_size=3, padding=1, stride=1)

    # ---------MobileV3Unet-------------
    # model = MobileV3Unet(num_classes=num_classes)
    # model.backbone['0'] = torch.nn.Sequential(
    #     nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    #     nn.BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
    #     nn.Hardswish()
    # )

    # ---------ResnetUnet-------------
    # model = Resnet18Unet(num_classes=num_classes)
    # model = Resnet34Unet(num_classes=num_classes)
    # model = Resnet50Unet(num_classes=num_classes)
    # model = Resnet101Unet(num_classes=num_classes)
    # model = Resnet152Unet(num_classes=num_classes)
    # model = Resnext50Unet(num_classes=num_classes)
    # model = Resnext101Unet(num_classes=num_classes)


    # ---------DensenetUnet121-------------
    # model = DensenetUnet(num_classes=num_classes)

    # ---------InceptionUnet-------------
    # model = InceptionUnet(num_classes=num_classes)

    # ---------EfficientnetUnet_b7-------------
    # model = EfficientnetUnet(num_classes=num_classes)
    
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    results_file = "resnet152.txt"

    train_dataset = DriveDataset(os.path.join(args.data_path, 'train'),
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std),
                                 view='default') # view:[default, multi-view1, multi-view2, multi-view3]

    val_dataset = DriveDataset(os.path.join(args.data_path, 'val'),
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std),
                               view='default')

    # test_dataset = DriveDataset(os.path.join(args.data_path, 'test'),
    #                            train=False,
    #                            transforms=get_transform(train=False, mean=mean, std=std),
    #                            view='default') 

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                          batch_size=1,
    #                                          num_workers=num_workers,
    #                                          pin_memory=True,
    #                                          collate_fn=test_dataset.collate_fn)

    model = create_model(num_classes=num_classes)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        # model.load_state_dict(checkpoint['model'], strict=False)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        # print(model.in_conv)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    
    # model = nn.DataParallel(model, device_ids=[2,3])
    model.to(device)
    model1 = None
    model2 = None
    model3 = None

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    best_dice = 0.
    start_time = time.time()
    losses = []
    train_dices = []
    val_dices = []
    for epoch in range(args.start_epoch, args.epochs):

        mean_loss, lr = train_one_epoch(model, model1, model2, model3, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        train_confmat, train_dice = evaluate(model, model1, model2, model3, train_loader, device=device, num_classes=num_classes)

        confmat, dice = evaluate(model, model1, model2, model3, val_loader, device=device, num_classes=num_classes)

        losses.append(mean_loss)
        train_dices.append(train_dice)
        val_dices.append(dice)
        np.savetxt('lossdice_resnet152.txt', (losses, train_dices, val_dices))

        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_info: \n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"train dice coefficient: {train_dice:.3f}\n"
            val_info = f"val_info: \n" \
                       f"val dice coefficient: {dice:.3f}\n"
            f.write(train_info + str(train_confmat) + "\n" + val_info + str(confmat) + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        
        torch.save(save_file, "save_weight/last_resnet152.pth")

        if args.save_best is True:
            torch.save(save_file, "save_weight/best_resnet152.pth")
        else:
            torch.save(save_file, "save_weight/model_{}.pth".format(epoch))


    # test_confmat, test_dice = evaluate(model, test_loader, device=device, num_classes=num_classes)
    test_confmat, test_dice = evaluate(model, model1, model2, model3, val_loader, device=device, num_classes=num_classes)
    with open(results_file, "a") as f:
            test_info = f"test_info: \n" \
                       f"test dice coefficient: {test_dice:.3f}\n"
            f.write(test_info + str(test_confmat) + "\n\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./data", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weight"):
        os.mkdir("./save_weight")

    main(args)


