import torch
import torch.nn as nn
import torch.optim as optim

from fbanks.cosine_dsitance_triplet_loss import TripletLoss


class ResBlock(nn.Module):

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


class FbanksVoiceNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_layer = TripletLoss(0.2)
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=(5 - 1)//2, stride=2),
            ResBlock(in_channels=64, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=(5 - 1)//2, stride=2),
            ResBlock(in_channels=128, out_channels=128, kernel_size=3),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            ResBlock(in_channels=256, out_channels=256, kernel_size=3),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            ResBlock(in_channels=512, out_channels=512, kernel_size=3),
            nn.AvgPool2d(kernel_size=4)
        )
        self.linear = nn.Linear(512, 256)

    def forward(self, anchor, positive, negative):
        n = anchor.shape[0]
        anchor_out = self.network(anchor)
        anchor_out = anchor_out.reshape(n, 512)
        anchor_out = self.linear(anchor_out)

        positive_out = self.network(positive)
        positive_out = positive_out.reshape(n, 512)
        positive_out = self.linear(positive_out)

        negative_out = self.network(negative)
        negative_out = negative_out.reshape(n, 512)
        negative_out = self.linear(negative_out)

        return anchor_out, positive_out, negative_out

    def loss(self, anchor, positive, negative, reduction='mean'):
        loss_val = self.loss_layer(anchor, positive, negative)
        return loss_val
