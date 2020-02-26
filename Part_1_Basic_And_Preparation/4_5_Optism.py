import torch as t
import torch.nn as nn
from torch import optim

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


# 定义LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x


net = LeNet().to(device)

# 定义优化器
optimizer = optim.SGD(params=net.parameters(), lr=1)
optimizer.zero_grad()  # 梯度清零，net.zero_grad()

input = t.randn(1, 3, 32, 32).to(device)
output = net(input)
output.backward(output)

# 执行优化
optimizer.step()

# 设定优化参数，学习率为1e-5
optimizer =optim.SGD([
                {'params': net.features.parameters()},
                {'params': net.classifier.parameters(), 'lr': 1e-2}
            ], lr=1e-5)

# 为特定层设定学习率，两个全连接层设置较大的学习率=0.01，其余层的学习率=0.001
special_layers = nn.ModuleList([net.classifier[0], net.classifier[3]])
special_layers_params = list(map(id, special_layers.parameters()))
base_params = filter(lambda p: id(p) not in special_layers_params,
                     net.parameters())

optimizer = t.optim.SGD([
            {'params': base_params},
            {'params': special_layers.parameters(), 'lr': 0.01}
        ], lr=0.001 )


# 调整学习率的两周方式
# 新建optimizer
old_lr = 0.1
optimizer1 =optim.SGD([
                {'params': net.features.parameters()},
                {'params': net.classifier.parameters(), 'lr': old_lr*0.1}
            ], lr=1e-5)
# 手动decay, 保存动量
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1


