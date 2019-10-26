# DepthNet
Monocular depth estimation with ResNet-50

## Data Sources
I collect most of the depth data for this project using [Depth Dector](https://github.com/Olament/DepthDetection), an iOS Application i created to obtain depth information via iDevices's TrueDepth camera.

### NYU Depth Dataset V2
The NYU-Depth V2 data set is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect.

## Model
### Architecture
![](https://github.com/Olament/DepthNet/blob/master/imgs/architecture.png)
*Architecuture figure from Iron Laina et al. paper*

The images are first feed into a ResNet-50 network and then go through four upsampling blocks. Different from what described in the figure, our model has an input size of  320x240x3 (WxHxC) and output size of 160x128x1 (WxHxC).

## Features to implement
- [ ] Data augmentation
- [ ] More loss functions
- [ ] Experiments on Upsample/Upconvolution block

## Citation
[Deep Optics for Monocular Depth Estimation and 3D Object Detection](https://arxiv.org/abs/1904.08601) by Julie Chang and Gordon Wetzstein

[Depth Estimation from Single Image Using CNN-Residual Network](http://cs231n.stanford.edu/reports/2017/pdfs/203.pdf) by Xiaobai Ma, Zhenglin Geng, and Zhi Bie

[Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/pdf/1606.00373.pdf) by Iro Laina
