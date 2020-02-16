import torch as t
import  matplotlib
from matplotlib import pyplot as plt

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

#手动设置随机种子
t.manual_seed(1000)

#y=x*5+3，掺噪声
def get_fake_data(batch_size=8):
    x = t.rand(batch_size, 1, device=device) * 5
    y = x * 5 + 2 + 3 * t.randn(batch_size, 1, device=device)
    y = y.to(device)
    return x, y

x, y = get_fake_data(batch_size=32)
plt.scatter(x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy())
#plt.show()

#随机初始化参数
w = t.rand(1, 1).to(device)
b = t.zeros(1, 1).to(device)

lr =0.02

for i in range(5000):
    x, y = get_fake_data(batch_size=4)

    #前向：计算loss
    y_pred = x.mm(w) + b.expand_as(y)  #torch.mm(a,b)返回a,b点积
    loss = 0.5 * (y_pred - y) ** 2     #方差
    loss = loss.mean()

    #反向：计算梯度
    dloss = 1
    dy_pred = dloss * (y_pred - y)

    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()

    # 更新参数
    w.sub_(lr * dw)                    #tensor.sub(value) 返回向量每个值减去Value后的向量
    b.sub_(lr * db)

    if i%500==0 :
        print("loss = %f"%loss.item())
#        print(w.item(),b.item())


# 画图
x = t.arange(0, 6).view(-1, 1).float().to(device)
y = x.mm(w).to(device) + b.expand_as(x).to(device)
x = x.cpu()
y = y.cpu()
print(x,y)

#预测图形与预测点
plt.plot(x.numpy(), y.numpy())
plt.scatter(x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy())
#绘图范围
#plt.xlim(0,5)
#plt.ylim(0,30)

plt.show()

print('w = ',w.item(),'b = ', b.item())
