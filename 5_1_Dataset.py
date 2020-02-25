import torch as t
from torch.utils import data
import os
from PIL import  Image
import numpy as np
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

#使用训练集Dogs vs. Cats

#归一化处理
transform = T.Compose([
    T.Resize(224),                                    #缩放图片(Image)，最短边为224像素
    T.CenterCrop(224),                                #切出224*224的图片
    T.ToTensor(),                                     #将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  #标准化至[-1, 1]，规定均值和标准差
])

class DogCat(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'dog' in img_path.split('/')[-1] else 1
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


dataset = DogCat(r"e:\TorchStudy\dogs-vs-cats\train", transforms=transform)
img, label = dataset[0]
#for img, label in dataset:
#    print(img.size(), label)

dataset = ImageFolder(r"e:\TorchStudy\dogs-vs-cats")
