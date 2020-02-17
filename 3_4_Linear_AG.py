import torch as t
from matplotlib import pyplot as plt
import numpy as np

t.manual_seed(1000)

def get_fake_data(batch_size=8):
    x = t.rand(batch_size,1) * 5
    y = x * 2 + 3 + t.randn(batch_size, 1)
    return x, y
