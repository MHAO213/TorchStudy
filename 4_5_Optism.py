import torch as t
import torch.nn as nn

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")