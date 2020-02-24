import torch as t
from torch.utils import data

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

#使用训练集Dogs vs. Cats

