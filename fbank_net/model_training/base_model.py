from abc import abstractmethod

from torch import nn


class FBankResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.network(x)
        out = out + x
        out = self.relu(out)
        return out


class FBankNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 - 1)//2, stride=2),
            FBankResBlock(in_channels=32, out_channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 - 1)//2, stride=2),
            FBankResBlock(in_channels=64, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            FBankResBlock(in_channels=128, out_channels=128, kernel_size=3),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            FBankResBlock(in_channels=256, out_channels=256, kernel_size=3),
            nn.AvgPool2d(kernel_size=4)
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(256, 250)
        )

    @abstractmethod
    def forward(self, *input_):
        raise NotImplementedError('Call one of the subclasses of this class')
