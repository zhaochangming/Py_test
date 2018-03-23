import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
import numpy as np

EPOCH = 10
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


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, hidden3):
        super(AutoEncoder, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, hidden3),
        )
        self.Decoder = nn.Sequential(
            nn.Linear(hidden3, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.Encoder(x)
        decoded = self.Decoder(encoded)
        return encoded, decoded


dae = AutoEncoder(28 * 28, 128, 64, 10)
optimizer = torch.optim.Adam(dae.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(EPOCH):
    train_loss = 0
    for step, (x, _) in enumerate(train_loader):
        x = Variable(x)

        _, decoded = dae(x)
        loss = criterion(decoded, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]

    print('Epoch: ', epoch, '| train loss: %.4f' % train_loss)

torch.save(dae.Encoder.state_dict(), 'dae.pth')
print(dae.Encoder[0].weight)