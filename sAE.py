import time
import argparse
import datetime
from pathlib import Path
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision import transforms
import numpy as np
from aeNet import stackedAE


def argumentInit():
    parser = argparse.ArgumentParser(
        description='training of Pytorch NN for MNIST dataset')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='input batch size for training (default:256)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='number of the epoch to train (default:400)')
    parser.add_argument(
        '--LR',
        type=float,
        default=0.1,
        help='initial learning rate for training (default:0.01)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGD momentum (default:0.9)')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='weight decay (default:0)')
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default=None,
        help='the path of pretrained model')
    parser.add_argument(
        '--milestones',
        type=list,
        default=[],
        help='epoch that reduce the learning rate (default:[every 80s])')
    parser.add_argument(
        'embedding_dimension',
        type=int,
        default=10,
        help='dimension of embedded feature (default:10)')
    parser.add_argument(
        'outdir_path', type=str, help='directory path of outputs')
    args = parser.parse_args()
    return args


def main(args):
    trans = transforms.ToTensor()
    trainDataSet = mnist.MNIST(
        './data', train=True, transform=trans, download=False)
    valDataSet = mnist.MNIST(
        './data', train=False, transform=trans, download=True)
    trainLoader = DataLoader(
        trainDataSet, batch_size=args.batch_size, shuffle=True)
    valLoader = DataLoader(
        valDataSet, batch_size=args.batch_size, shuffle=True)
    loader = {'train': trainLoader, 'val': valLoader}
    #dataset_sizes = {'train': len(trainDataSet), 'val': len(valDataSet)}
    print("Complete the preparing dataset")
    w = h = 28
    net = stackedAE(w * h, args.embedding_dimension)
    net.initialization()
    print("Initialized the network parameters")

    for num, stack in enumerate(net.stacks):
        pretrainLayers = [
            net.coder[num].parameters(),
            net.coder[-1 * (num + 1)].parameters()
        ]
        print('Init stackNum: {}'.format(num + 1))
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(
            [
                {
                    'params': pretrainLayers[0]
                },
                {
                    'params': pretrainLayers[1]
                },
            ],
            lr=0.1,
            momentum=0.9,
            weight_decay=0)

        for epoch in range(5):
            running_loss = 0.0
            for i, (inputs, _) in enumerate(trainLoader):
                inputs = inputs.view(-1, w * h)
                inputs = Variable(inputs)
                optimizer.zero_grad()
                outputs = net(inputs)[2][num]
                loss = criterion(outputs, inputs)
                running_loss += loss.data[0]
                loss.backward()
                optimizer.step()
            epoch_loss = running_loss / len(trainLoader)
            print('epoch: {},epoch_loss:{}'.format(epoch + 1, epoch_loss))

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.LR,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    loss_history = {"train": [], "val": []}
    start_time = time.time()

    for epoch in range(args.epochs):
        print('* ' * 20)
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('* ' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            running_loss = 0.0
            for i, (inputs, _) in enumerate(loader[phase]):
                inputs = inputs.view(-1, w * h)

                if phase == 'train':
                    inputs = Variable(inputs)
                else:
                    inputs = Variable(inputs, volatile=True)

                optimizer.zero_grad()
                _, outputs, _ = net(inputs)
                loss = criterion(outputs, inputs)
                running_loss += loss.data[0]

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            epoch_loss = running_loss / len(loader[phase])
            loss_history[phase].append(epoch_loss)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        elapsed_time // 60, elapsed_time % 60))
    return net, loss_history


def write_parameters(args):
    import csv
    fout = open(
        Path(args.outdir_path).joinpath('experimental_settings.csv'), "wt")
    csvout = csv.writer(fout)
    print('*' * 50)
    print('Parameters')
    print('*' * 50)
    for arg in dir(args):
        if not arg.startswith('_'):
            csvout.writerow([arg, str(getattr(args, arg))])
            print('%-25s %-25s' % (arg, str(getattr(args, arg))))


if __name__ == '__main__':
    args = argumentInit()
    write_parameters(args)

    model_weights, loss_history = main(args)
    torch.save(model_weights.state_dict(),
               Path(args.outdir_path).joinpath('weight.pth'))
    training_history = np.zeros((2, args.epochs))
    for i, phase in enumerate(["train", "val"]):
        training_history[i] = loss_history[phase]
    np.save(
        Path(args.outdir_path).joinpath('training_history_{}.npy'.format(
            datetime.date.today())), training_history)
