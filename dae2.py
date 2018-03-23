import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision import transforms
import numpy as np

EPOCH = 30
BATCH_SIZE = 30000
LR = 0.005
DOWMLOAD_MNIST = False


trans = transforms.ToTensor()


train_set = mnist.MNIST(
    './data', train=True, transform=trans, download=False)

train_loader = DataLoader(
    dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)


class dae(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, hidden3):
        super(dae, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
        )
        self.cla = nn.Sequential(nn.Linear(10, 10))

    def forward(self, x):
        x = self.Encoder(x)
        y_pre = self.cla(x)
        return y_pre

    def initialization(self, mean=0, std=0.01):
        nn.init.normal(self.Encoder[0].weight, mean, std)
        nn.init.normal(self.Encoder[2].weight, mean, std)
        nn.init.normal(self.Encoder[4].weight, mean, std)
        nn.init.normal(self.cla[0].weight, mean, std)


net = dae(28 * 28, 500, 200, 10)
net.initialization()
net.Encoder.load_state_dict(torch.load('weight.pth'))

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    train_loss = 0
    for step, (x, y) in enumerate(train_loader):
        x = Variable(x.view(-1, 28 * 28))
        y = Variable(y)

        y_pre = net(x)
        loss = criterion(y_pre, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]

    train_loss = train_loss / len(train_loader)
    print('Epoch: ', epoch, '| train loss: %.4f' % train_loss)
