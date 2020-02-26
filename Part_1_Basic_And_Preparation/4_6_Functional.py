import torch as t
import torch.nn as nn

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

#使用ReLU与Linear进行测试
input = t.randn(2, 3).to(device)
model1 = nn.Linear(3, 4).to(device)
output1 = model1(input)
output2 = nn.functional.relu(input)
print(input,output1,output2)

from torch.nn import functional as F

# nn.functional与nn.Module混用
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.pool(F.relu(self.conv1(x)), 2)
        x = F.pool(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x