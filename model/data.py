import torch
import os
from skimage import io
from skimage.transform import resize
from PIL import Image
import torchvision.transforms as transforms


class DepthDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.path = "data.nosync/"
        self.lst = [f for f in sorted(os.listdir(self.path)) if "r-" in f]

    def __getitem__(self, index):
        # load image
        image = Image.open(os.path.join(self.path, self.lst[index]))
        depth = resize(io.imread(os.path.join(self.path, self.lst[index]).replace("r-", "d-")), (62, 78))

        # transformation
        image_trans = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
        ])
        depth_trans = transforms.Compose([
            transforms.ToTensor()
        ])
        image = image_trans(image)
        depth = depth_trans(depth)

        return image, depth.float()

    def __len__(self):
        return len(self.lst)

