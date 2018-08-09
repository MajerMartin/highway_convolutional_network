import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayCNN(nn.Module):
    """
    Highway convolutional network based on
    https://arxiv.org/pdf/1505.00387.pdf
    """
    def __init__(self, channels=32, kernel_size=3, depth=3, bias=-1, max_pool=True):
        super(HighwayCNN, self).__init__()
        self.depth = depth
        self.kernel_size = kernel_size
        self.max_pool = max_pool

        # calculate padding to preserve output shape of conv layers
        padding = tuple((k - 1) // 2 for k in (kernel_size, kernel_size))
        
        # calculate flatten dimensions
        img_dim = 28
        max_pool_kernel_size = 2

        if max_pool:
            # for padding = 0, dilation = 1
            dim = ((img_dim - (max_pool_kernel_size - 1) - 1) // max_pool_kernel_size) + 1
        else:
            # for dilation = 1, stride = 1
            dim = img_dim + 2 * padding[0] - (kernel_size - 1)
        
        self.conv = nn.Conv2d(1, channels, kernel_size=kernel_size,
                              padding=padding)
        self.conv_H = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=kernel_size,
                      padding=padding) for _ in range(depth)
        ])
        self.conv_T = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=kernel_size,
                      padding=padding) for _ in range(depth)
        ])
        self.max_pool = nn.MaxPool2d(max_pool_kernel_size)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(channels * dim * dim, 64)
        self.fc2 = nn.Linear(64, 10)

        # initialize the bias for the carry gates
        for i in range(depth):
            self.conv_T[i].bias.data.fill_(bias)

    def forward(self, x):
        x = F.relu(self.conv(x))

        for i in range(self.depth):
            H = F.relu(self.conv_H[i](x))
            T = torch.sigmoid(self.conv_T[i](x))
            x = H * T + x * (1 - T)
            
            # TODO: add max pooling to every highway block
            x = self.dropout(x)
        
        if self.max_pool:
            x = self.max_pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x