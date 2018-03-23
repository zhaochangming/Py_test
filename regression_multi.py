import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

w_target = np.array([0.5, 3, 2.4])
b_target = np.array([0.9])
f = 'y = {:.2f} + {:.2f}*X + {:.2f}*X^2 +{:.2f}*X^3'.format(
    b_target[0], w_target[0], w_target[1], w_target[2])
print(f)

x_sample = np.arange(-3, 3.1, 0.1)
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample**2 + w_target[2] * x_sample**3
plt.plot(x_sample, y_sample, 'b')
plt.show()

x_train = np.stack([x_sample**i for i in range(1, 4)], axis=1)
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_sample).float().unsqueeze(1)

w = Variable(torch.randn(3, 1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)

x_train = Variable(x_train)
y_train = Variable(y_train)


def multi_linear(x):
    return torch.mm(x, w) + b


y = multi_linear(x_train)

plt.plot(x_train.data.numpy()[:, 0], y.data.numpy(), 'r')
plt.plot(x_train.data.numpy()[:, 0], y_sample, 'b')
plt.show()


def get_loss(y_t, y_p):
    return torch.mean((y_t - y_p)**2)


loss = get_loss(y, y_train)
loss.backward()

for e in range(100):
    y = multi_linear(x_train)
    loss = get_loss(y, y_train)

    w.grad.zero_()
    b.grad.zero_()
    loss.backward()

    w.data = w.data - 1e-3 * w.grad.data
    b.data = b.data - 1e-3 * b.grad.data
    if (e + 1) % 20 == 0:
        print('epoch{},loss:{:.5f}'.format(e + 1, loss.data[0]))
y = multi_linear(x_train)

plt.plot(x_train.data.numpy()[:, 0], y.data.numpy(), 'r')
plt.plot(x_train.data.numpy()[:, 0], y_sample, 'b')
plt.show()
