import torch, torchvision
import torch.nn as nn
from collections import OrderedDict

import torch
import torch.nn as nn


# 定义CBAM注意力模块
class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# 定义通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        output = self.sigmoid(avg_out + max_out)
        return output * x

# 定义空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        output = self.sigmoid(out)
        return output * x



class Resnet18FPN(nn.Module):
    def __init__(self):
        super(Resnet18FPN, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]
        self.conv5 = children[7]

    def forward(self, im_data):
        feat = OrderedDict()
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        feat_map5 = self.conv5(feat_map4)
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        feat['map5'] = feat_map5
        return feat


class DensityRegressor(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(DensityRegressor, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            # nn.Conv2d(input_channels, 196, 7, padding=3),
            # nn.ReLU(),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Conv2d(196, 128, 5, padding=2),
            # nn.ReLU(),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Conv2d(128, 64, 3, padding=1),
            # nn.ReLU(),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Conv2d(64, 32, 1),
            # nn.ReLU(),
            # nn.Conv2d(32, 1, 1),
            # nn.ReLU(),
            nn.ConvTranspose2d(input_channels, 196, 4, stride=2, padding=1),  # 使用反卷积替代上采样
            nn.ReLU(),
            nn.ConvTranspose2d(196, 128, 4, stride=2, padding=1),  # 使用反卷积替代上采样
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 使用反卷积替代上采样
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 使用反卷积替代上采样
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU()
        )

    def forward(self, im):
        num_sample = im.shape[0]
        if num_sample == 1:
            output = self.regressor(im.squeeze(0))
            if self.pool == 'mean':
                output = torch.mean(output, dim=(0), keepdim=True)
                return output
            elif self.pool == 'max':
                output, _ = torch.max(output, 0, keepdim=True)
                return output
        else:
            for i in range(0, num_sample):
                output = self.regressor(im[i])
                if self.pool == 'mean':
                    output = torch.mean(output, dim=(0), keepdim=True)
                elif self.pool == 'max':
                    output, _ = torch.max(output, 0, keepdim=True)
                if i == 0:
                    Output = output
                else:
                    Output = torch.cat((Output, output), dim=0)
            return Output


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def weights_xavier_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
