import torch as t
import torch.nn as nn

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.register_parameter('param1',nn.Parameter(t.randn(3, 3)))
        #self.param1 = nn.Parameter(t.rand(3, 3))与上行等价
        self.submodel1 = nn.Linear(3, 4)
    def forward(self, input):
        x = self.param1.mm(input)
        x = self.submodel1(x)
        return x
net = Net()

#输出net的参数
print(net)
print(net._modules)
print(net._parameters)
print(net.param1)

for name, param in net.named_parameters():
    print(name, param.size())
#submodel为子模型
for name, submodel in net.named_modules():
    print(name, submodel)


#保存模型
t.save(net.state_dict(), 'net.pth')

#加载模型
net2 = Net()
net2.load_state_dict(t.load('net.pth'))

