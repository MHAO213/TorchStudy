from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as t
import torchvision


#导入MNIST数据集
dataset = datasets.MNIST(root = "E:\TorchStudy\MNIST",transform=transforms,train = True)

#Tensor-->IMG
to_pil = transforms.ToPILImage()
tadokoro = to_pil(t.rand(3, 114, 514))
#tadokoro.show()

print(len(dataset))

dataloader = DataLoader(dataset, shuffle=True, batch_size=16)

#图片拼合
dataiter = iter(dataloader)
img = torchvision.utils.make_grid(next(dataiter)[0], 4)      #Error:'module' object is not callable

to_img = transforms.ToPILImage()
OUT = to_img(img)
OUT.show()