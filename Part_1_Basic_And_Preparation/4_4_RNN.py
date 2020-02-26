import torch as t
import torch.nn as nn

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

t.manual_seed(1000)

#输入：batch_size=3，序列长度为2，每个元素占5维
input = t.randn(2, 3, 5)

#LSTM
#lstm输入向量5维，隐藏元3，1层
lstm = nn.LSTM(5, 3, 1)

#初始状态：1层，batch_size=3，3个隐藏元
h0 = t.randn(1, 3, 3)
c0 = t.randn(1, 3, 3)
out, hn = lstm(input, (h0, c0))

#LSTM CELL
#一个LSTMCell对应的层数只能是一层
input = t.randn(2, 3, 4)
lstm = nn.LSTMCell(4, 3)
hx = t.randn(3, 3)
cx = t.randn(3, 3)
out = [] in input:
    hx, cx=lstm(i, (hx, cx))
    out.append(hx)
t.stack(out)

#有4个词，每个词用5维的向量表示
embedding = nn.Embedding(4, 5)
#预训练词向量初始化embedding
embedding.weight.data = t.arange(0,20).view(4,5)
input = t.arange(3, 0, -1).long()
output = embedding(input)