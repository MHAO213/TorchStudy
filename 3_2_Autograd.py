import torch as t

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

#几种不同制定requires_grad的方式
a = t.ones(3,3,requires_grad=True,dtype=float).to(device)
b = t.zeros_like(a)
#a.requires_grad=True
#a = t.zeros(3,4).requires_grad_()

c = a + b
d = c.sum()
d.backward()
#print(a.grad)

##定义函数f(x)，手动给定导函数gradf(x)，与反向传播结果对比
def f(x):
    y = x**2 * t.exp(x) + t.sin(x)
    return y

def gradf(x):
    dx = 2*x*t.exp(x) + x**2*t.exp(x) + t.cos(x)
    return dx

x = t.randn(3,4).to(device)
x.requires_grad_()                          #求梯度与定义分开写 否则梯度为none
y = f(x).to(device)
y.backward(t.ones(y.size()).to(device))
#print(x.grad,gradf(x))