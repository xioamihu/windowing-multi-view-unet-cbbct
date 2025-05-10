from collections import OrderedDict
from typing import Dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
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


class EfficientnetUnet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False):
        super(EfficientnetUnet, self).__init__()
        backbone = efficientnet_b7(pretrained=False)

        # backbone.load_state_dict(torch.load("../save_weights/inception_v3_google-0cc3c7bd.pth", map_location='cpu'))

        backbone = backbone.features

        # self.stage_out_channels = [16, 24, 40, 112, 320]  # b0,b1
        # self.stage_out_channels = [16, 24, 48, 120, 352]  # b2
        # self.stage_out_channels = [24, 32, 48, 136, 384]  # b3
        # self.stage_out_channels = [24, 32, 56, 160, 448]  # b4
        # self.stage_out_channels = [24, 40, 64, 176, 512]  # b5
        # self.stage_out_channels = [32, 40, 72, 200, 576]  # b6
        self.stage_out_channels = [32, 48, 80, 224, 640]  # b7

        self.backbone = IntermediateLayerGetter(backbone, {'1': 'stage0', '2': 'stage1', '3': 'stage2', '5': 'stage3', '7': 'stage4'})

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
        x = self.up1(backbone_out['stage4'], backbone_out['stage3'])
        x = self.up2(x, backbone_out['stage2'])
        x = self.up3(x, backbone_out['stage1'])
        x = self.up4(x, backbone_out['stage0'])
        x = self.conv(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return {"out": x}


# backbone = efficientnet_b0(pretrained=False).features
# # backbone.AuxLogits = nn.Sequential()
# # print(backbone)
# x = torch.randn(4, 3, 224, 224)
# # print(backbone(x).shape)
# # new_m = IntermediateLayerGetter(backbone, {'0': 'stage0', '1': 'stage1', '2': 'stage2', '3': 'stage3', '4': 'stage4',
# #                                            '5': 'stage5', '6': 'stage6', '7': 'stage7'})
# new_m = IntermediateLayerGetter(backbone, {'1': 'stage0', '2': 'stage1', '3': 'stage2', '5': 'stage3', '7': 'stage4'})
# out = new_m(x)
# print([(k, v.shape) for k, v in out.items()])
# model = EfficientnetUnet(num_classes=2)
# # print(model)
# print(model(x)['out'].shape)
