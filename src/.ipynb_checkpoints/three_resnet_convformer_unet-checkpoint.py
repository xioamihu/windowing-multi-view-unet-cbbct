import numpy as np
from collections import OrderedDict
from typing import Dict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, \
    wide_resnet50_2, wide_resnet101_2
from src.unet import Up, OutConv


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def relative_pos_dis(height=32, weight=32, sita=0.9):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2
    #  dis = torch.exp(-dis*(1/(2*sita**2)))
    return dis


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


class CNNAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_patches=1024):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_patches = num_patches

        # self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, padding=0, bias=False)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=3, padding=1, bias=False)
        self.dis = relative_pos_dis(math.sqrt(num_patches), math.sqrt(num_patches), sita=0.9)#.to("cuda:1")
        self.headsita = nn.Parameter(torch.randn(heads), requires_grad=True)
        self.sig = nn.Sigmoid()

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),  # inner_dim
            nn.ReLU(inplace=True),
        ) if project_out else nn.Identity()

    def forward(self, x, mode="train", smooth=1e-4):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (g d) h w -> b g (h w) d', g=self.heads), qkv)
        attn = torch.matmul(q, k.transpose(-1, -2))  # b g n n
        qk_norm = torch.sqrt(torch.sum(q ** 2, dim=-1)+smooth)[:, :, :, None] * torch.sqrt(torch.sum(k ** 2, dim=-1)+smooth)[:, :, None, :] + smooth
        attn = attn/qk_norm
        #  attentionheatmap_visual2(attn, self.sig(self.headsita), out_dir='./Visualization/ACDC/SETR_plane2', value=1)
        #  factor = 1/(2*(self.sig(self.headsita)+0.01)**2) # h
        factor = 1/(2*(self.sig(self.headsita)*(0.4-0.003)+0.003)**2)  # af3 + limited setting this, or using the above line code
        dis = factor[:, None, None]*(self.dis[None, :, :].to(factor.device))  # g n n
        dis = torch.exp(-dis)
        dis = dis/torch.sum(dis, dim=-1)[:, :, None]
        #  attentionheatmap_visual2(dis[None, :, :, :], self.sig(self.headsita), out_dir='./Visualization/ACDC/dis', value=0.003)
        # attn = attn * dis[None, :, :, :]
        _, _, h, w = attn.shape
        attn = attn * dis[None, :, 0:h, 0:w]
        #  attentionheatmap_visual2(attn, self.sig(self.headsita), out_dir='./Visualization/ACDC/after', value=0.003)
        #  attentionheatmap_visual(attn, out_dir='./Visualization/attention_af3/')
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b g (h w) d -> b (g d) h w', h=x.shape[2])
        if mode=="train":
            return self.to_out(out)
        else:
            return self.to_out(out), attn


class CNNFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class CNNTransformer_record(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=1024):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CNNAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches),
                CNNFeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ThreeResnetConvFormerUnet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ThreeResnetConvFormerUnet, self).__init__()
        backbone = resnext50_32x4d(pretrained=pretrained)
        backbone1 = resnet34(pretrained=pretrained)
        backbone2 = resnet152(pretrained=pretrained)

        # backbone.load_state_dict(torch.load("./save_weights/resnet34-333f7ec4.pth", map_location='cpu'), strict=False)
        # backbone1.load_state_dict(torch.load("./save_weights/resnet18-5c106cde.pth", map_location='cpu'), strict=False)
        # backbone2.load_state_dict(torch.load("./save_weights/resnet18-5c106cde.pth", map_location='cpu'), strict=False)

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
            in_channels=self.stage_out_channels[4] + self.stage_out_channels1[4] + self.stage_out_channels2[4],
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

        self.transformer = CNNTransformer_record(dim=2048, depth=12, heads=8, dim_head=64, mlp_dim=4*2048, dropout=0.1, num_patches=1024)

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
        x4 = self.convconcate1(
            torch.cat((backbone_out['stage4'], backbone_out1['stage4'], backbone_out2['stage4']), dim=1))

        x4 = self.transformer(x4)

        x3 = self.convconcate2(
            torch.cat((backbone_out['stage3'], backbone_out1['stage3'], backbone_out2['stage3']), dim=1))
        x2 = self.convconcate3(
            torch.cat((backbone_out['stage2'], backbone_out1['stage2'], backbone_out2['stage2']), dim=1))
        x1 = self.convconcate4(
            torch.cat((backbone_out['stage1'], backbone_out1['stage1'], backbone_out2['stage1']), dim=1))
        x0 = self.convconcate5(
            torch.cat((backbone_out['stage0'], backbone_out1['stage0'], backbone_out2['stage0']), dim=1))

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


class up_attention(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_attention, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.conv = nn.Sequential(
            nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.attention_spatial = AttentionSpatial(F_g=out_ch, F_l=out_ch, F_int=out_ch)
        self.att_channel = AttentionChannel(channel=out_ch*2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x1 = self.up_conv(x1)
        x2 = self.attention_spatial(x1, x2)  # d,e

        x = torch.cat([x2, x1], dim=1)
        x = self.att_channel(x)
        x = self.conv(x)
        return x


class AttentionChannel(nn.Module):
    # channel-wise attention
    def __init__(self, channel, reduction=4, multiply=True):
        super(AttentionChannel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.multiply = multiply

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.multiply:
            return x * y
        else:
            return y


class AttentionSpatial(nn.Module):
    """
    Attention Block
    spatial attention
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionSpatial, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):   # g, x = d, e
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class ThreeResnetConvFormerUnetAttn(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ThreeResnetConvFormerUnetAttn, self).__init__()
        backbone = resnext50_32x4d(pretrained=pretrained)
        backbone1 = resnet34(pretrained=pretrained)
        backbone2 = resnet152(pretrained=pretrained)

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
            in_channels=self.stage_out_channels[4] + self.stage_out_channels1[4] + self.stage_out_channels2[4],
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

        self.transformer = CNNTransformer_record(dim=2048, depth=12, heads=8, dim_head=64, mlp_dim=4*2048, dropout=0.1, num_patches=1024)

        self.up_attn1 = up_attention(in_ch=self.stage_out_channels[4], out_ch=self.stage_out_channels[3])
        self.up_attn2 = up_attention(in_ch=self.stage_out_channels[3], out_ch=self.stage_out_channels[2])
        self.up_attn3 = up_attention(in_ch=self.stage_out_channels[2], out_ch=self.stage_out_channels[1])
        self.up_attn4 = up_attention(in_ch=self.stage_out_channels[1], out_ch=self.stage_out_channels[0])

        self.conv = OutConv(self.stage_out_channels[0], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        backbone_out = self.backbone(x)
        backbone_out1 = self.backbone1(x)
        backbone_out2 = self.backbone2(x)

        # 'stage4' ---> 'stage0' 的特征融合
        # print(backbone_out['stage4'].shape, backbone_out1['stage4'].shape, backbone_out2['stage4'].shape)
        x4 = self.convconcate1(
            torch.cat((backbone_out['stage4'], backbone_out1['stage4'], backbone_out2['stage4']), dim=1))

        x4 = self.transformer(x4)

        x3 = self.convconcate2(
            torch.cat((backbone_out['stage3'], backbone_out1['stage3'], backbone_out2['stage3']), dim=1))
        x2 = self.convconcate3(
            torch.cat((backbone_out['stage2'], backbone_out1['stage2'], backbone_out2['stage2']), dim=1))
        x1 = self.convconcate4(
            torch.cat((backbone_out['stage1'], backbone_out1['stage1'], backbone_out2['stage1']), dim=1))
        x0 = self.convconcate5(
            torch.cat((backbone_out['stage0'], backbone_out1['stage0'], backbone_out2['stage0']), dim=1))

        x = self.up_attn1(x4, x3)
        x = self.up_attn2(x, x2)
        x = self.up_attn3(x, x1)
        x = self.up_attn4(x, x0)

        x = self.conv(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return {"out": x}


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ThreeResnetUnet(num_classes=2)
# # model.load_state_dict(torch.load("./save_weights/resnet34-333f7ec4.pth", map_location='cpu'), strict=False)
# model.to(device)
# x = torch.randn(1, 3, 121, 125).to(device)    # b x c x h x w
# y = model(x)
# # print(model)
# print(y["out"].shape)


# # model.out_conv = nn.Sequential()
# # y = model(x)
# # print(model)
# # print(y["out"].shape)


# checkpoint = torch.load("E:/pengdie/predict/result2/ResNet/ResNet34Aug/best_weight0814.pth", map_location='cpu')
# # print(checkpoint)
# # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# print(checkpoint["model"]['backbone'])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ThreeResnetUnet(num_classes=2)
# backbone1_weights = torch.load("./save_weights/last_weight0824.pth", map_location='cpu')['model']
# # print('Loading base network...')
# new_state_dict1 = OrderedDict()
# for k, v in backbone1_weights.items():
#     name = k[9:]  # remove `vgg.`，即只取vgg.0.weights的后面几位
#     new_state_dict1[name] = v
#     model.backbone1.load_state_dict(new_state_dict1)
#
# backbone2_weights = torch.load("./save_weights/last_resnet152.pth.pth", map_location='cpu')['model']
# # print('Loading base network...')
# new_state_dict2 = OrderedDict()
# for k, v in backbone2_weights.items():
#     name = k[9:]  # remove `vgg.`，即只取vgg.0.weights的后面几位
#     new_state_dict2[name] = v
#     model.backbone2.load_state_dict(new_state_dict2)
#
# # model.backbone.load_state_dict(torch.load("./save_weights/last_weight0824.pth", map_location='cpu')['model'], strict=False)
# # model.backbone1.load_state_dict(torch.load("./save_weights/last_weight0814.pth", map_location='cpu')['model'], strict=False)
# # model.backbone2.load_state_dict(torch.load("./save_weights/last_resnet152.pth", map_location='cpu')['model'], strict=False)
# model.load_state_dict(torch.load("./save_weights/last_weight0824.pth", map_location='cpu'), strict=False)
# model.to(device)
# x = torch.randn(1, 3, 512, 512).to(device)    # b x c x h x w
# y = model(x)
# print(model)
# print(y["out"].shape)
# # model.out_conv = nn.Sequential()
# # y = model(x)
# # print(model)
# # print(y["out"].shape)


