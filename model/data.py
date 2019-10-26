import torch
import os
from PIL import Image
import torchvision.transforms as transforms


class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.lst = []
        for r, d, f in os.walk(path):
            for file in f:
                if 'rgb' in file:
                    self.lst.append(os.path.join(r, file))

    def __getitem__(self, index):
        # load image
        image = Image.open(self.lst[index])
        depth = Image.open(self.lst[index].replace("rgb", "depth"))

        # transformation
        comm_trans = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.CenterCrop((228, 304)),
            transforms.RandomHorizontalFlip()
        ])
        image_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        depth_trans = transforms.Compose([
            transforms.Resize((64, 80)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()),
            #transforms.Lambda(lambda x:  torch.div(x, 65535.0))
        ])
        image = image_trans(comm_trans(image))
        depth = depth_trans(comm_trans(depth))

        return image, depth

    def __len__(self):
        return len(self.lst)



