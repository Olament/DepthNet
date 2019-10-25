import torch
import os
from PIL import Image
import torchvision.transforms as transforms


class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.lst = [f for f in sorted(os.listdir(self.path)) if "rgb" in f]

    def __getitem__(self, index):
        # load image
        image = Image.open(os.path.join(self.path, self.lst[index]))
        depth = Image.open(os.path.join(self.path, self.lst[index].replace("rgb", "depth")))

        # transformation
        image_trans = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        depth_trans = transforms.Compose([
            transforms.Resize((128, 160)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5,), (0.5,))
        ])
        image = image_trans(image)
        depth = depth_trans(depth)

        return image, depth.float()

    def __len__(self):
        return len(self.lst)