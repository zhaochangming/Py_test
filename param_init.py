import numpy as np
import torch
from torch import nn

net1 = nn.Sequential(
    nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 1))

w1 = net1[0].weight
print(w1)
w2 = net1[2].weight
print(w2)
w3 = net1[4].weight
print(w3)

net1[0].weight.data = torch.from_numpy(np.random.uniform(3, 4, size=(4, 3)))

w1 = net1[0].weight
print(w1)
w2 = net1[2].weight
print(w2)
w3 = net1[4].weight
print(w3)

for layer in net1:
    if isinstance(layer, nn.Linear):  # 判断是否是线性层
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(
            np.random.normal(0, 0.5, size=param_shape))

w1 = net1[0].weight
print(w1)
w2 = net1[2].weight
print(w2)
w3 = net1[4].weight
print(w3)