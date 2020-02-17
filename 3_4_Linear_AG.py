import torch as t
from matplotlib import pyplot as plt
import numpy as np

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

t.manual_seed(1000)

def get_fake_data(batch_size=16):
    x = t.rand(batch_size,1,device=device) * 5
    y = x * 5 + 3 + 2 * t.randn(batch_size, 1,device=device)
    return x, y

x, y = get_fake_data()
plt.scatter(x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy())
#plt.show()

w = t.rand(1, 1, requires_grad=True, device=device)
b = t.zeros(1, 1, requires_grad=True,device=device)
losses = np.zeros(500)

lr = 0.005

for i in range(500):
    x, y = get_fake_data(batch_size=32)

    #前向计算loss
    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    losses[i] = loss.item()

    #反向计算梯度
    loss.backward()

    #更新w b
    w.data.sub_(lr * w.grad.data)
    b.data.sub_(lr * b.grad.data)

    #梯度清零
    w.grad.data.zero_()
    b.grad.data.zero_()

x = t.arange(0, 6).view(-1, 1).float().to(device)
y = x.mm(w.data).to(device) + b.expand_as(x).to(device)
x = x.cpu()
y = y.cpu()
print(x,y)

x = x.data
y = y.data

#预测图形与预测点
plt.plot(x.numpy(), y.numpy())
plt.scatter(x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy())
#绘图范围
#plt.xlim(0,5)
#plt.ylim(0,30)

plt.show()

print(w.item(), b.item())