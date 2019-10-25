# DepthNet
Monocular depth estimation with CNN

## Data Sources
I collect most of the depth for this project using [Depth Dector](https://github.com/Olament/DepthDetection), an iOS Application i created to obtain depth information via iDevices's TrueDepth camera.

## Model
### Architecture
![](https://github.com/Olament/DepthNet/blob/master/imgs/architecture.png)
*Architecuture figure from Iron Laina et al. paper*

The images are first feed into a ResNet-50 network and then go through four upsampling blocks. Different from what described in the figure, our model has an input size of  320x240x3 (WxHxC) and output size of 78x62x1 (WxHxC).

## Citation
[Deep Optics for Monocular Depth Estimation and 3D Object Detection](https://arxiv.org/abs/1904.08601) by Julie Chang and Gordon Wetzstein

[Depth Estimation from Single Image Using CNN-Residual Network](http://cs231n.stanford.edu/reports/2017/pdfs/203.pdf) by Xiaobai Ma, Zhenglin Geng, and Zhi Bie

[Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/pdf/1606.00373.pdf) by Iro Laina
