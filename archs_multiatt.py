import torch
from torch import nn


from convnext import convnext_tiny




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x



class squeeze_excitation_block(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(in_c, out_c, kernel_size=1, padding=0)
        )

        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0, dilation=1)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(in_c, out_c, kernel_size=3, padding=18, dilation=18)

        self.c5 = Conv2D(out_c * 5, out_c, kernel_size=1, padding=0, dilation=1)

    def forward(self, x):
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=x.size()[2:], mode="bilinear", align_corners=True)

        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)

        xc = torch.cat([x0, x1, x2, x3, x4], axis=1)
        y = self.c5(xc)

        return y
class ASPP_Convnext(nn.Module):
    def __init__(self, in_c, mid_c):
        super().__init__()

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(in_c, mid_c, kernel_size=1, padding=0)
        )

        self.c1 = Conv2D(in_c, mid_c, kernel_size=1, padding=0, dilation=1)
        self.c2 = Conv2D(in_c, mid_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, mid_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(in_c, mid_c, kernel_size=3, padding=18, dilation=18)

        self.c5 = Conv2D(mid_c * 5, in_c, kernel_size=1, padding=0, dilation=1)

    def forward(self, x):
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=x.size()[2:], mode="bilinear", align_corners=True)

        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)

        xc = torch.cat([x0, x1, x2, x3, x4], axis=1)
        y = self.c5(xc)

        return y

class CACB(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        # self.c1 =fasterblock.Faster_Block()
        self.c1 = Conv2D1x1(in_c, out_c)
        # self.c2 = Conv2D(out_c, out_c)
        self.c2 =Block(out_c)
        # self.a1 = squeeze_excitation_block(out_c)
        self.a2 = CBAM(out_c)
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.a2(x)
        # x = self.a1(x)

        return x


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c, out_c)
        self.a1 = squeeze_excitation_block(out_c)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.a1(x)
        return x

class VGGBlock19(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


from typing import  List
class Permute(torch.nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.permute(x, self.dims)

from functools import partial



from timm.models.layers import  DropPath

class yencoder1(nn.Module):
    def __init__(self):
        super().__init__()
        nb_filter = [64, 128, 256, 512]
        # network = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # network = vgg19()
        # network = convnext_base()
        # 0-7
        # print(network.features)
        # self.x1 = network.features[:4]
        # self.x2 = network.features[4:9]
        # self.x3 = network.features[9:18]
        # self.x4 = network.features[18:27]
        # self.x5 = network.features[27:36]
        # self.conv0_0 = CNBlock(3, nb_filter[0])
        # self.conv1_0 = CNBlock(nb_filter[0], nb_filter[1])
        # self.conv2_0 = CNBlock(nb_filter[1], nb_filter[2])
        # self.conv3_0 = CNBlock(nb_filter[2], nb_filter[3])
        # self.conv4_0 = CNBlock(nb_filter[3], nb_filter[3])

        # print(network)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = VGGBlock19(3, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock19(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock19(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock19(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock19(nb_filter[3], nb_filter[3], nb_filter[3])
        # network = vgg19()
        # self.x1 = network.features[:4]
        # self.x2 = network.features[4:9]
        # self.x3 = network.features[9:18]
        # self.x4 = network.features[18:27]
        # self.x5 = network.features[27:36]
        # #
        # self.ca = ChannelAttention(64)
        # self.ca1 = ChannelAttention(128)
        # self.ca2 = ChannelAttention(256)
        # self.ca3 = ChannelAttention(512)
        # self.ca4 = ChannelAttention(512)
        # self.sa = SpatialAttention()

    def forward(self, x):
        x0 = x
        x1 = self.conv0_0(x0)
        # # # 64  256
        x2 = self.conv1_0(self.pool(x1))
        # # # 128  128
        x3 = self.conv2_0(self.pool(x2))
        # # # 256   64
        x4 = self.conv3_0(self.pool(x3))
        # # # 512   32
        x5 = self.conv4_0(self.pool(x4))
        # 512   16
        # x1 = self.x1(x0)
        # x1 = self.sa(self.ca(x1) * x1) * x1
        #
        # x2 = self.x2(x1)
        # x2 = self.sa(self.ca1(x2) * x2) * x2
        #
        # x3 = self.x3(x2)
        # x3 = self.sa(self.ca2(x3) * x3) * x3
        #
        # x4 = self.x4(x3)
        # x4 = self.sa(self.ca3(x4) * x4) * x4
        #
        # x5 = self.x5(x4)
        # x5 = self.sa(self.ca4(x5) * x5) * x5
        return x5, [x4, x3, x2, x1]
        # return  x1
class zencoder(nn.Module):
    def __init__(self):
        super().__init__()
        nb_filter = [96, 192, 384, 768]
        self.convnext =convnext_tiny(pretrained=True)

        # self.convnext = ConvNeXt()
        #
        # self.cbam1 = CBAM(nb_filter[0])
        # self.cbam2 = CBAM(nb_filter[1])
        # self.cbam3 = CBAM(nb_filter[2])
        # self.cbam4 = CBAM(nb_filter[3])

    def forward(self, x):
        x0 = x

        x1 = self.convnext.stages[0](self.convnext.downsample_layers[0](x))
        x2 = self.convnext.stages[1](self.convnext.downsample_layers[1](x1))
        x3 = self.convnext.stages[2](self.convnext.downsample_layers[2](x2))
        x4 = self.convnext.stages[3](self.convnext.downsample_layers[3](x3))

        # x1 = self.conv0_0(x0)
        # # # 64  256
        # x2 = self.conv1_0(self.pool(x1))
        # # # 128  128
        # x3 = self.conv2_0(self.pool(x2))
        # # # 256   64
        # x4 = self.conv3_0(self.pool(x3))
        # # # 512   32
        # x5 = self.conv4_0(self.pool(x4))
        # 512   16
        # x1 = self.x1(x0)
        # x1 = self.sa(self.ca(x1) * x1) * x1
        #
        # x2 = self.x2(x1)
        # x2 = self.sa(self.ca1(x2) * x2) * x2
        #
        # x3 = self.x3(x2)
        # x3 = self.sa(self.ca2(x3) * x3) * x3
        #
        # x4 = self.x4(x3)
        # x4 = self.sa(self.ca3(x4) * x4) * x4
        #
        # x5 = self.x5(x4)
        # x5 = self.sa(self.ca4(x5) * x5) * x5
        return x4, [x3, x2, x1]
        # return  x1





class encoder1(nn.Module):
    def __init__(self):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = VGGBlock(3, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        return x4_0, [x3_0, x2_0, x1_0, x0_0]



class ydecoder1(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(64 + 512, 256)
        self.c2 = conv_block(512, 128)
        self.c3 = conv_block(256, 64)
        self.c4 = conv_block(128, 32)

    def forward(self, x, skip):
        s1, s2, s3, s4 = skip

        x = self.up(x)
        x = torch.cat([x, s1], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, s2], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, s3], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, s4], axis=1)
        x = self.c4(x)

        return x

class zdecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up1 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.c1 = conv_block(64 + 384, 256)
        self.c2 = conv_block(256+192, 128)
        self.c3 = conv_block(128+96, 64)
        self.c4 = conv_block(64, 1)

    def forward(self, x, skip):
        s1, s2, s3 = skip

        x = self.up(x)
        x = torch.cat([x, s1], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, s2], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, s3], axis=1)
        x = self.c3(x)

        # x = self.up(x)
        # x = torch.cat([x, s4], axis=1)
        # x = self.c4(x)

        return self.up1(x)

class zencoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))

        self.c1 = conv_block(3, 32)
        self.c2 = conv_block(32, 64)
        self.c3 = conv_block(64, 128)
        self.c4 = conv_block(128, 256)

    def forward(self, x):
        x0 = x

        x1 = self.c1(x0)
        p1 = self.pool(x1)

        x2 = self.c2(p1)
        p2 = self.pool(x2)

        x3 = self.c3(p2)
        p3 = self.pool(x3)

        x4 = self.c4(p3)
        p4 = self.pool(x4)

        return p4, [x4, x3, x2, x1]

class yencoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))

        self.c1 = conv_block(3, 32)
        self.c2 = conv_block(32, 64)
        self.c3 = conv_block(64, 128)
        self.c4 = conv_block(128, 256)

    def forward(self, x):
        x0 = x

        x1 = self.c1(x0)
        p1 = self.pool(x1)

        x2 = self.c2(p1)
        p2 = self.pool(x2)

        x3 = self.c3(p2)
        p3 = self.pool(x3)

        x4 = self.c4(p3)
        p4 = self.pool(x4)

        return p4, [x4, x3, x2, x1]


class ydecoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(832, 256)
        self.c2 = conv_block(640, 128)
        self.c3 = conv_block(320, 64)
        self.c4 = conv_block(160, 32)

    def forward(self, x, skip1, skip2):
        x = self.up(x)
        x = torch.cat([x, skip1[0], skip2[0]], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, skip1[1], skip2[1]], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, skip1[2], skip2[2]], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, skip1[3], skip2[3]], axis=1)
        x = self.c4(x)

        return x
class zdecoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(384+64+256, 256)
        self.c2 = conv_block(192+128+256, 128)
        self.c3 = conv_block(128+64+96, 64)
        self.c4 = conv_block(64+32, 32)
        # self.c1 = conv_block(384+64+512, 256)
        # self.c2 = conv_block(192+256+256, 128)
        # self.c3 = conv_block(128+128+96, 64)
        # self.c4 = conv_block(64+64, 32)

    def forward(self, x, skip1, skip2):
        x = self.up(x)
        x = torch.cat([x, self.up(skip1[0]), skip2[0]], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, self.up(skip1[1]), skip2[1]], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, self.up(skip1[2]), skip2[2]], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x,  skip2[3]], axis=1)
        x = self.c4(x)

        return x

class decoder1(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(64 + 256, 256)
        self.c2 = conv_block(256 + 128, 128)
        self.c3 = conv_block(128 + 64, 64)
        self.c4 = conv_block(64 + 32, 32)

    def forward(self, x, skip):
        s1, s2, s3, s4 = skip

        x = self.up(x)
        x = torch.cat([x, s1], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, s2], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, s3], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, s4], axis=1)
        x = self.c4(x)

        return x


class encoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))
        self.c1 = conv_block(3, 32)
        self.c2 = conv_block(32, 64)
        self.c3 = conv_block(64, 128)
        self.c4 = conv_block(128, 256)

    def forward(self, x):
        x0 = x

        x1 = self.c1(x0)
        # 32 256 256
        p1 = self.pool(x1)
        # 32 128 128
        x2 = self.c2(p1)
        # 64 128 128
        p2 = self.pool(x2)
        # 64 64 64
        x3 = self.c3(p2)
        # 128 64 64
        p3 = self.pool(x3)
        # 128 32 32
        x4 = self.c4(p3)
        # 246 32 32
        p4 = self.pool(x4)
        # 256 16 16
        return p4, [x4, x3, x2, x1]





class decoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.c1 = conv_block(832, 256)
        # self.c2 = conv_block(640, 128)
        # self.c3 = conv_block(320, 64)
        # self.c4 = conv_block(160, 32)
        self.c1 = conv_block(64 + 256 + 256, 256)
        self.c2 = conv_block(256 + 128 + 128, 128)
        self.c3 = conv_block(128 + 64 + 64, 64)
        self.c4 = conv_block(32 + 32 + 64, 32)

    def forward(self, x, skip1, skip2):
        x = self.up(x)
        x = torch.cat([x, skip1[0], skip2[0]], axis=1)
        #  640 64 64
        x = self.c1(x)
        #  128 128 128
        x = self.up(x)
        # 320 128 128
        x = torch.cat([x, skip1[1], skip2[1]], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, skip1[2], skip2[2]], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, skip1[3], skip2[3]], axis=1)
        x = self.c4(x)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class Extraction(nn.Module):
    """
    iteration
    """

    def __init__(self, A_now, B, channels):
        super(Extraction, self).__init__()
        self.A_now = A_now
        self.B = B
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        u, f = x

        x = F.relu(self.bn1(f - self.A_now(u)))
        x = F.relu(self.bn2(self.B(x)) + u)

        x = (x, f)
        return x


class Conv_block_left(nn.Module):

    def __init__(self, A_now, channels):
        super(Conv_block_left, self).__init__()
        B_conv_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_1 = Extraction(A_now, B_conv_1, channels)

        B_conv_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_2 = Extraction(A_now, B_conv_2, channels)

        B_conv_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_3 = Extraction(A_now, B_conv_3, channels)

    def forward(self, u_f):
        u_f = self.extraction_1(u_f)
        u_f = self.extraction_2(u_f)
        u_f = self.extraction_3(u_f)
        return u_f  # new_u and f


class Restriction(nn.Module):

    def __init__(self, A_now, A_next, channels):
        super(Restriction, self).__init__()
        self.Pai = nn.Conv2d(channels, channels, 3, 2, 1, bias=False)
        self.A_now = A_now
        self.A_next = A_next

        self.R = nn.Conv2d(channels, channels, 3, 2, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, out):
        u, f = out
        del out
        # Pool u
        u_next = self.Pai(u)

        # update f. follow:  pool(f) - pool( a_now(u) ) + a_next( pool(u_next) )
        f = self.R(f - self.A_now(u)) + self.A_next(u_next)
        del u
        f = self.relu(f)
        f = self.bn(f)

        return (u_next, f)


class Conv_block_right(nn.Module):

    def __init__(self, A_now, channels):
        super(Conv_block_right, self).__init__()
        B_conv_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_1 = Extraction(A_now, B_conv_1, channels)

        B_conv_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_2 = Extraction(A_now, B_conv_2, channels)

        B_conv_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_3 = Extraction(A_now, B_conv_3, channels)

        B_conv_4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_4 = Extraction(A_now, B_conv_4, channels)

    def forward(self, u_f):
        u_f = self.extraction_1(u_f)
        u_f = self.extraction_2(u_f)
        u_f = self.extraction_3(u_f)
        u_f = self.extraction_4(u_f)
        return u_f[0]


class Conv_block_last(nn.Module):

    def __init__(self, A_now, channels):
        super(Conv_block_last, self).__init__()
        B_conv_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_1 = Extraction(A_now, B_conv_1, channels)

        B_conv_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_2 = Extraction(A_now, B_conv_2, channels)

        B_conv_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_3 = Extraction(A_now, B_conv_3, channels)

        B_conv_4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_4 = Extraction(A_now, B_conv_4, channels)

        B_conv_5 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_5 = Extraction(A_now, B_conv_5, channels)

        B_conv_6 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_6 = Extraction(A_now, B_conv_6, channels)

        B_conv_7 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_7 = Extraction(A_now, B_conv_7, channels)

    def forward(self, u_f):
        u_f = self.extraction_1(u_f)
        u_f = self.extraction_2(u_f)
        u_f = self.extraction_3(u_f)
        u_f = self.extraction_4(u_f)
        u_f = self.extraction_5(u_f)
        u_f = self.extraction_6(u_f)
        u_f = self.extraction_7(u_f)

        return u_f[0]


class Right_u_init(nn.Module):

    def __init__(self, channels):
        super(Right_u_init, self).__init__()
        self.upsample = nn.ConvTranspose2d(channels, channels,
                                           kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, coarseU_initU_refineU):
        # error
        out = coarseU_initU_refineU[0] - coarseU_initU_refineU[1]
        out = self.upsample(out)

        # u + u_0
        out = coarseU_initU_refineU[2] + out

        return out

def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block
class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h, w), mode='bilinear', align_corners=True)
def HASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 1), padding=(atrous_rate, 0),
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block
def VASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, atrous_rate),
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block
class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer):
        super(ASPP_Module, self).__init__()
        # In our re-implementation of ASPP module,
        # we follow the original paper but change the output channel
        # from 256 to 512 in all of four branches.
        # out_channels = in_channels // 4
        out_channels = 64

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.V = VASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.H = HASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.V1 = VASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.H1 = HASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.V2 = VASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.H2 = HASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.cross_ver = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1),
                      stride=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.cross_hor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0),
                      stride=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        # self.cross_ver = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), stride=1)
        # self.cross_hor = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer)
        # self.convaspp = VGGBlock1(out_channels * 5, in_channels, in_channels)
        # self.convaspp1 = VGGBlock1(out_channels * 8, in_channels, in_channels)
        self.convaspp = Conv2D(out_channels * 5, 64)
        self.convaspp1 = Conv2D(out_channels * 8, 64)
        # self.convaspp1 = Conv2D(out_channels * 13, 64)

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        cross_ver = self.V(x)
        cross_ver2 = self.V1(x)
        cross_ver3 = self.V2(x)
        cross_hor2 = self.H(x)
        cross_hor = self.H1(x)
        cross_hor3 = self.H2(x)
        cross_ver1 = self.cross_ver(x)
        cross_hor1 = self.cross_hor(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        y1 = torch.cat((cross_ver, cross_hor, cross_ver1, cross_hor1, cross_ver2, cross_hor2, cross_ver3, cross_hor3), 1)
        y = self.convaspp(y)
        y1 = self.convaspp1(y1)
        # return y + y1
        # y1 = torch.cat((feat0, feat1, feat2, feat3, feat4, cross_ver, cross_hor, cross_ver1, cross_hor1, cross_ver2, cross_hor2, cross_ver3, cross_hor3), 1)
        # y1 = self.convaspp1(y1)
        return y+y1
        # return y

import torch
import numpy as np
from torch import nn
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, inputChannel, outputChannel, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(inputChannel, outputChannel, stride)
        self.bn1 = nn.BatchNorm2d(outputChannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outputChannel, outputChannel)
        self.bn2 = nn.BatchNorm2d(outputChannel)
        self.downsample = downsample
        self.ca = ChannelAttention(outputChannel)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        caOutput = self.ca(out)
        out = caOutput * out
        saOutput = self.sa(out)
        out = saOutput * out
        return out, saOutput


class BasicDownSample(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(inputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.LeakyReLU(0.2),
            nn.Conv2d(outputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        x = self.convolution(x)
        return x


