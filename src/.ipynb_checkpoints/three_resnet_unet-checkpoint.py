import numpy as np
from collections import OrderedDict
from typing import Dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from src.unet import Up, OutConv


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ThreeResnetUnet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ThreeResnetUnet, self).__init__()
        backbone = resnext50_32x4d(pretrained=pretrained)
        backbone1 = resnet34(pretrained=pretrained)
        backbone2 = resnet152(pretrained=pretrained)

        # weights_path1 = "./save_weights/last_weight0824.pth"
        # weights_path2 = "./save_weights/last_weight0814.pth"
        # weights_path3 = "./save_weights/last_resnet152.pth"

        # backbone.load_state_dict(torch.load("./save_weights/last_weight0824.pth", map_location='cpu')['model'], strict=False)
        # backbone1.load_state_dict(torch.load("./save_weights/last_weight0814.pth", map_location='cpu')['model'], strict=False)
        # backbone2.load_state_dict(torch.load("./save_weights/last_resnet152.pth", map_location='cpu')['model'], strict=False)

        # self.stage_out_channels = stage_out_channels
        # self.stage_out_channels = [64, 64, 128, 256, 512]  # resnet18 和 resnet34

        # resnet50 和 resnet101 和 resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
        # self.stage_out_channels = [64, 256, 512, 1024, 2048]

        self.stage_out_channels = [64, 256, 512, 1024, 2048]
        self.stage_out_channels1 = [64, 64, 128, 256, 512]
        self.stage_out_channels2 = [64, 256, 512, 1024, 2048]

        # resnet18 和 resnet34 :conv1 ,,,,, 其余  relu
        self.backbone = IntermediateLayerGetter(backbone, {'relu': 'stage0', 'layer1': 'stage1', 'layer2': 'stage2',
                                                           'layer3': 'stage3', 'layer4': 'stage4'})

        self.backbone1 = IntermediateLayerGetter(backbone1, {'relu': 'stage0', 'layer1': 'stage1', 'layer2': 'stage2',
                                                             'layer3': 'stage3', 'layer4': 'stage4'})

        self.backbone2 = IntermediateLayerGetter(backbone2, {'relu': 'stage0', 'layer1': 'stage1', 'layer2': 'stage2',
                                                             'layer3': 'stage3', 'layer4': 'stage4'})

        self.convconcate1 = DoubleConv(
            in_channels=self.stage_out_channels[4]+self.stage_out_channels1[4]+self.stage_out_channels2[4],
            out_channels=self.stage_out_channels[4])

        self.convconcate2 = DoubleConv(
            in_channels=self.stage_out_channels[3] + self.stage_out_channels1[3] + self.stage_out_channels2[3],
            out_channels=self.stage_out_channels[3])

        self.convconcate3 = DoubleConv(
            in_channels=self.stage_out_channels[2] + self.stage_out_channels1[2] + self.stage_out_channels2[2],
            out_channels=self.stage_out_channels[2])

        self.convconcate4 = DoubleConv(
            in_channels=self.stage_out_channels[1] + self.stage_out_channels1[1] + self.stage_out_channels2[1],
            out_channels=self.stage_out_channels[1])

        self.convconcate5 = DoubleConv(
            in_channels=self.stage_out_channels[0] + self.stage_out_channels1[0] + self.stage_out_channels2[0],
            out_channels=self.stage_out_channels[0])

        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])
        self.conv = OutConv(self.stage_out_channels[0], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        backbone_out = self.backbone(x)
        backbone_out1 = self.backbone1(x)
        backbone_out2 = self.backbone2(x)

        # 'stage4' ---> 'stage0' 的特征融合
        # print(backbone_out['stage4'].shape, backbone_out1['stage4'].shape, backbone_out2['stage4'].shape)
        x4 = self.convconcate1(torch.cat((backbone_out['stage4'], backbone_out1['stage4'], backbone_out2['stage4']), dim=1))
        x3 = self.convconcate2(torch.cat((backbone_out['stage3'], backbone_out1['stage3'], backbone_out2['stage3']), dim=1))
        x2 = self.convconcate3(torch.cat((backbone_out['stage2'], backbone_out1['stage2'], backbone_out2['stage2']), dim=1))
        x1 = self.convconcate4(torch.cat((backbone_out['stage1'], backbone_out1['stage1'], backbone_out2['stage1']), dim=1))
        x0 = self.convconcate5(torch.cat((backbone_out['stage0'], backbone_out1['stage0'], backbone_out2['stage0']), dim=1))

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        # x = self.up1(backbone_out['stage4'], backbone_out['stage3'])
        # x = self.up2(x, backbone_out['stage2'])
        # x = self.up3(x, backbone_out['stage1'])
        # x = self.up4(x, backbone_out['stage0'])

        x = self.conv(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return {"out": x}


# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# model = ThreeResnetUnet(num_classes=2)

# backbone1_weights = torch.load("./save_weights/last_weight0814.pth", map_location='cpu')['model']
# new_state_dict1 = OrderedDict()
# for k, v in backbone1_weights.items():
#     if k[:8] == 'backbone':
#         name = k[9:]  # remove `vgg.`，即只取vgg.0.weights的后面几位
#         # print(k)
#         # print(name)
#         new_state_dict1[name] = v
# model.backbone1.load_state_dict(new_state_dict1)

# backbone2_weights = torch.load("./save_weights/last_resnet152.pth", map_location='cpu')['model']
# # print('Loading base network...')
# new_state_dict2 = OrderedDict()
# for k, v in backbone2_weights.items():
#     if k[:8] == 'backbone':
#         name = k[9:]  # remove `vgg.`，即只取vgg.0.weights的后面几位
#         # print(k)
#         # print(name)
#         new_state_dict2[name] = v
# model.backbone2.load_state_dict(new_state_dict2)

# # # model.backbone.load_state_dict(torch.load("./save_weights/last_weight0824.pth", map_location='cpu')['model'], strict=False)
# # # model.backbone1.load_state_dict(torch.load("./save_weights/last_weight0814.pth", map_location='cpu')['model'], strict=False)
# # # model.backbone2.load_state_dict(torch.load("./save_weights/last_resnet152.pth", map_location='cpu')['model'], strict=False)
# model.load_state_dict(torch.load("./save_weights/last_weight0824.pth", map_location='cpu')['model'], strict=False)
# for name, para in model.named_parameters(): # .parameters()
#     # print(name)
#     if "convconcate" in name:
#         para.requires_grad = True
#     else:
#         para.requires_grad = False

# for name, para in model.named_parameters(): # .parameters()
#     if para.requires_grad == True:
#         print(name)

# model.to(device)
# x = torch.randn(4, 3, 512, 512).to(device)    # b x c x h x w
# y = model(x)
# # print(model)
# print(y["out"].shape)


