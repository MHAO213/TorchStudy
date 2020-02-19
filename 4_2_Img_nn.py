import torch as t
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

#读取图片
to_pil = ToPILImage()
bqb = Image.open(r'HG.png')
#bqb.show()

#图片转Tensor
to_tensor = ToTensor()
input = to_tensor(bqb).unsqueeze(0)

#锐化的卷积核
kernel = t.ones(3, 3)/-9.
kernel[1][1] = 1
#print(kernel)
conv = nn.Conv2d(4, 4, (3, 3), bias=False).to(device)         #RGBA通道的PNG具有4通道
conv.weight.data = (kernel.view(1,1,3,3).expand(1,4,3,3))     #tensor.view等同于remap expand扩充维度至与输入同维
#print(list(conv.parameters()))                               #卷积层参数为卷积核 上一行为指定卷积核参数

out = conv(input)
HG_C_OUT = to_pil(out.data.squeeze(0))
#HG_OUT.show()

#池化层
pool = nn.AvgPool2d(3,3).to(device)
out = pool(input)
#print(list(pool.parameters()))                               #池化层无参数
HG_P_OUT = to_pil(out.data.squeeze(0))
#HG_P_OUT.show()

#规范批化层
batchnorm = nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True)
print(list(batchnorm.parameters()))
out = batchnorm(input)
HG_B_OUT = to_pil(out.data.squeeze(0))
print(out.size())

