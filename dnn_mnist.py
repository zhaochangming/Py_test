import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import mnist

EPOCH = 10
BATCH_SIZE = 100
LR = 0.1


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.reshape(-1)
    x = torch.from_numpy(x)
    return x


train_set = mnist.MNIST(
    './data', train=True, transform=data_tf, download=False)
test_set = mnist.MNIST(
    './data', train=False, transform=data_tf, download=False)

train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_data = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
print(train_set[0][0].size())
net = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

for epoch in range(EPOCH):

    train_loss = 0
    train_acc = 0
    net.train()
    for step, (im, label) in enumerate(train_data):
        im = Variable(im)
        label = Variable(label)

        out = net(im)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]

        _, pred = out.max(1)
        num_correct = (pred == label).sum().data[0]
        acc = num_correct / im.shape[0]
        train_acc += acc
    train_loss = train_loss / len(train_data)
    train_acc = train_acc / len(train_data)

    eval_loss = 0
    eval_acc = 0
    net.eval()
    for step, (im, label) in enumerate(test_data):
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)

        eval_loss += loss.data[0]

        _, pred = out.max(1)
        num_correct = (pred == label).sum().data[0]
        acc = num_correct / im.shape[0]
        eval_acc += acc

    eval_loss = eval_loss / len(test_data)
    eval_acc = eval_acc / len(test_data)
    print(
        'epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
        .format(epoch, train_loss, train_acc, eval_loss, eval_acc))
