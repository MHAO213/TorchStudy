import torch as t
import torch.nn as nn
import numpy

#定义Tensor
x=t.ones(3,3)
y=t.rand_like(x)
z=y[1][1]

#Tensor NumPy互转
a=t.ones(5)
b=a.numpy()
c=t.from_numpy(b)
d=c[2].item()
#print(a.size(),d)

#CUDA加速
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(x.device)
u = x + y
#print(u)

#AutoGrad计算
x.requires_grad=True
y=t.ones_like(x,requires_grad=True)
y=x.sum()
y.backward()
#print(x,x.grad)
#print(y)
y.backward()
#print(x.grad)

#梯度清零 否则累加
x.grad.data.zero_()
y.backward()
#print(x.grad)

x.grad.data.zero_()
z=x+x
z=z+x
z=z.sum()
z.backward()
#print(z,x.grad)

q=t.tensor(0,dtype=t.float32)
q.requires_grad=True
w=t.ones_like(q)
e=10*w+q
r=e*3
r.backward()
print(q.grad)
#r==(e*3)==(10*w+q)*3 对q求导结果为3