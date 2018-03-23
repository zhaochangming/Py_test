import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
import numpy as np

EPOCH = 3
BATCH_SIZE = 30000
LR = 0.005
DOWMLOAD_MNIST = False


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = x.reshape(-1)
    x = torch.from_numpy(x)
    return x


train_set = mnist.MNIST(
    './data', train=True, transform=data_tf, download=False)

train_loader = DataLoader(
    dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)


class dae(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, hidden3):
        super(dae, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, hidden3),
        )

    def forward(self, x):
        y_pre = self.Encoder(x)
        return y_pre


net = dae(28 * 28, 128, 64, 10)
net.Encoder.load_state_dict(torch.load('dae.pth'))
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    train_loss = 0
    for step, (x, y) in enumerate(train_loader):
        x = Variable(x)
        y = Variable(y)

        y_pre = net(x)
        loss = criterion(y_pre, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]

    print('Epoch: ', epoch, '| train loss: %.4f' % train_loss)
