import torch as t
import torch.nn as nn

#继承nn.Module实现全连接层，nn.Parameter为网络中可学习参数
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        nn.Module.__init__(self)                                   #与super(Linear, self).__init__()等价
        self.w = nn.Parameter(t.randn(in_features, out_features))  #w,b为网络学习参数
        self.b = nn.Parameter(t.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b.expand_as(x)

#给定全连接层nn.Module(a,b)中a，b分别为输入、输出列向量个数，input与output维度一致
layer = Linear(4,5)
input = t.randn(2,4)
output = layer(input)
#print(input,output)

#查看网络参数
#for name, parameter in layer.named_parameters():
#    print(name, parameter)

#多层网络实现
class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        nn.Module.__init__(self)
        self.layer1 = Linear(in_features, hidden_features)         #Linear是前面定义的全连接层
        self.layer2 = Linear(hidden_features, out_features)        #学习参数的继承：即此处网络中Linear学习参数w,b继承自Linear
    def forward(self,x):
        x = self.layer1(x)
        x = t.sigmoid(x)
        return self.layer2(x)

#给定全连接nn.Module(a, b, c)中，a、b、c分别为输入、中间层、输出节点个数（对应(self, in_features, hidden_features, out_features)）
perceptron = Perceptron(3,6,2)
for name, param in perceptron.named_parameters():
    print(name, param.size())
