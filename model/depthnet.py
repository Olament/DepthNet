import torch
import torch.nn as nn
import torchvision

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        resnet = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(1024)
        # Add new upsampling layer
        self.up1 = nn.Sequential(UpSamplingBlock(1024, 512),
                                 UpSamplingBlock(512, 256),
                                 UpSamplingBlock(256, 128))
        self.conv2 = conv3x3(128, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        out = self.resnet(inputs)
        out = self.conv1(out)
        out = self.bn(out)
        out = self.up1(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out

model = DepthNet().to(device)
input = torch.rand(1, 3, 240, 320)
print(model(input.to(device)).size())