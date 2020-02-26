import torch as t

#device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

#只有对variable的操作才能使用autograd
a = t.ones(1)
b = t.rand(1)
c = t.rand(1)
b.requires_grad_()
c.requires_grad_()
x = a * c                                     #requires_grad有传递性 此处x.requires_grad值为True
y = x + b

#计算图本质上是上述算数展开 即y=x+b=a*c+b的一颗树
#print(a.is_leaf,b.is_leaf,c.is_leaf)
#print(x.is_leaf,y.is_leaf)

#tensor.grad本质是查看其传播函数，加法(y)为AddBackward，常数(b)为none
#print(y.grad_fn,b.grad_fn)

#tensor.grad_fn.next_functions是反向传播至前一变量的传播函数，[0][0]是变量x的位置，所以是x的传播函数（链式法则）
#其中x为乘法为MulBackward
#print(y.grad_fn.next_functions,x.grad_fn)
#print(x.grad_fn.next_functions)

#x由a、c两变量构成，a不需求导所以传播函数为none，c为常数，梯度累加
#print(x.grad_fn.next_functions)

#x反向传播中能看到c拥有标识AccumulateGrad，多次计算时梯度累加
#y.backward(retain_graph=True)                 #retain_graph=True保证计算梯度后中间变量不清空
#print(c.grad,x.grad_fn.next_functions[1])
#y.backward()
#print(c.grad)

#使用数据又不需要计算梯度可用Tensor.data inplace操作不能针对叶子节点进行
#print(b.requires_grad,b.data.requires_grad,b.data.sin_().requires_grad) #b.sin_()报错

#求特定梯度，调用函数torch.autograd.grad(y,x)，即y对x求导，包含反向传播
#print(t.autograd.grad(y, x))
