import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#归一化处理
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

#数据集准备
data_train = datasets.MNIST(root = "E:\TorchStudy\MNIST",
                            transform=trans,
                            train = True)

data_test = datasets.MNIST(root="E:\TorchStudy\MNIST",
                           transform = trans,
                           train = False)

#数据集装载
batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=data_train,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=data_test,
                batch_size=batch_size,
                shuffle=False)

#网络搭建 单通道输入 两层卷积 池化(conv1)后x.view扁平化压缩 全连接同时防过拟合(dropout=0.5)
class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x

#生成网络 损失函数与优化器
model = Net().to(device)
cost = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

n_epochs = 2
for epoch in range(n_epochs):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = cost(out, target)
        ave_loss = ave_loss * 0.9 + loss.item() * 0.1
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print("epoch: {}, batch index: {}, train loss: {:.6f}".format(epoch, batch_idx + 1, ave_loss))

    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = cost(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.item() * 0.1

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
            print("epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}".format(epoch, batch_idx + 1, ave_loss, correct_cnt * 1.0 / total_cnt))

