import torch.nn as nn


class stackedAE(nn.Module):
    def __init__(self, inputSize, embedding_dim):
        super(stackedAE, self).__init__()

        self.e1 = nn.Linear(inputSize, 500)
        self.e2 = nn.Linear(500, 200)
        self.e3 = nn.Linear(200, embedding_dim)

        self.d1 = nn.Linear(500, inputSize)
        self.d2 = nn.Linear(200, 500)
        self.d3 = nn.Linear(embedding_dim, 200)

        self.encoder = nn.Sequential(self.e1, nn.ReLU(), self.e2, nn.ReLU(),
                                     self.e3)
        self.decoder = nn.Sequential(self.d3, nn.ReLU(), self.d2, nn.ReLU(),
                                     self.d1)
        self.stack1 = nn.Sequential(
            self.e1, nn.ReLU(), nn.Dropout(p=0.2), self.d1)
        self.stack2 = nn.Sequential(
            self.e1,
            nn.ReLU(),
            nn.Dropout(p=0.2),
            self.e2,
            nn.ReLU(),
            nn.Dropout(p=0.2),
            self.d2,
            nn.ReLU(),
            nn.Dropout(p=0.2),
            self.d1)
        self.stack3 = nn.Sequential(self.encoder, self.decoder)
        self.stacks = [self.stack1, self.stack2, self.stack3]
        self.coder = [self.e1, self.e2, self.e3, self.d3, self.d2, self.d1]

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        stack_output = []
        for stack in self.stacks:
            stack_output.append(stack(x))

        return encoded, decoded, stack_output

    def initialization(self, mean=0, std=0.01):
        for layer in self.coder:
            nn.init.normal(layer.weight, mean, std)
