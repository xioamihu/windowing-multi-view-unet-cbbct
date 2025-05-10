import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d
from collections import OrderedDict
from typing import Dict
from torch import Tensor
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
        self.dis = relative_pos_dis(math.sqrt(num_patches), math.sqrt(num_patches), sita=0.9).cuda()
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
        dis = factor[:, None, None]*self.dis[None, :, :]  # g n n
        dis = torch.exp(-dis)
        dis = dis/torch.sum(dis, dim=-1)[:, :, None]
        #  attentionheatmap_visual2(dis[None, :, :, :], self.sig(self.headsita), out_dir='./Visualization/ACDC/dis', value=0.003)
        attn = attn * dis[None, :, :, :]
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


class Res_ConvFormer(nn.Module):
    def __init__(self, n_channels, n_classes, imgsize, patch_num=32, dim=512, depth=12, heads=8, mlp_dim=512*4, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(imgsize)
        self.patch_height, self.patch_width = pair(imgsize//patch_num)
        self.dmodel = dim

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num

        # self.cnn_encoder = CNNEncoder2(n_channels, dim, self.patch_height, self.patch_width)  # the original is CNNs
        backbone = resnext50_32x4d(pretrained=False)
        self.cnn_encoder = IntermediateLayerGetter(backbone, {'layer4': 'stage4'})
        # self.cnn_encoder = resnet34(pretrained=False)  # resnext50_32x4d

        self.transformer = CNNTransformer_record(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel//4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.cnn_encoder(img)['stage4']
        # encoder
        x = self.transformer(x)  # b c h w -> b c h w
        x = self.decoder(x)
        return {"out": x}


class Res_ConvFormer_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, imgsize, patch_num=32, dim=512, depth=12, heads=8, mlp_dim=512*4, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(imgsize)
        self.patch_height, self.patch_width = pair(imgsize//patch_num)
        self.dmodel = dim

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num

        # self.cnn_encoder = CNNEncoder2(n_channels, dim, self.patch_height, self.patch_width)  # the original is CNNs
        self.stage_out_channels = [64, 256, 512, 1024, 2048]
        backbone = resnext50_32x4d(pretrained=False)
        self.cnn_encoder = IntermediateLayerGetter(backbone, {'relu': 'stage0', 'layer1': 'stage1', 'layer2': 'stage2',
                                                              'layer3': 'stage3', 'layer4': 'stage4'})

        self.transformer = CNNTransformer_record(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)

        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])
        self.conv = OutConv(self.stage_out_channels[0], num_classes=n_classes)

    def forward(self, img):
        input_shape = img.shape[-2:]
        backbone_out = self.cnn_encoder(img)
        # encoder
        x = self.transformer(backbone_out['stage4'])  # b c h w -> b c h w

        # x = self.decoder(x)
        x = self.up1(x, backbone_out['stage3'])
        x = self.up2(x, backbone_out['stage2'])
        x = self.up3(x, backbone_out['stage1'])
        x = self.up4(x, backbone_out['stage0'])

        x = self.conv(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return {"out": x}


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# imgsize = 512
# # model = Setr_ConvFormer(n_channels=3, n_classes=2, imgsize=256, patch_num=32,
# #                         dim=512, depth=12, heads=8, mlp_dim=512*4, dim_head=64, dropout=0.1)

# # model = Res_ConvFormer(n_channels=3, n_classes=2, imgsize=imgsize, patch_num=imgsize//32,
# #                        dim=2048, depth=12, heads=8, mlp_dim=2048*4, dim_head=64, dropout=0.1)

# model = Res_ConvFormer_UNet(n_channels=3, n_classes=2, imgsize=imgsize, patch_num=imgsize//32,
#                             dim=2048, depth=12, heads=8, mlp_dim=2048*4, dim_head=64, dropout=0.1)
# model.to(device)
# x = torch.randn(1, 3, imgsize, imgsize).to(device)
# print(model(x)['out'].shape)
