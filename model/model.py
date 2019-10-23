import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Datasets and loader
dataset = torchvision.datasets.CIFAR10(root='../../data',
                                       transform=transforms.ToTensor())

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=64,
                                     shuffle=False)


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
# 5x5 convolution
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)

# UpSampling Block
class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSamplingBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.conv1 = conv5x5(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        # create indices for unpool
        size = x.size()
        _, indices = self.pool(torch.empty(size[0], size[1], size[2]*2, size[3]*2))
        # unpool and assign residual
        out = self.unpool(x, indices.to(device))
        residual = self.conv1(out)
        # forward and projection
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


# DepthNet
class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        # Remove FC and AvgPool layer from Resnet34
        resnet = torchvision.models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Add new upsampling layer

    def forward(self, inputs):
        outputs = self.resnet(inputs)
        return outputs

model = UpSamplingBlock(1024, 512).to(device)

# def get_output_size(m, i, o):
#     print(m)
#     print(o.size())
#     print()
#
# for name, layer in model._modules.items():
#     layer.register_forward_hook(get_output_size)

with torch.no_grad():
    image = torch.rand(1, 1024, 8, 10)
    print(image.size())
    output = model(image.to(device))
    print(output.size())


