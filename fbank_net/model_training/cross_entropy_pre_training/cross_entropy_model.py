from torch import nn

from fbank_net.model_training.base_model import FBankNet


class FBankCrossEntropyNet(FBankNet):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_layer = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, x):
        n = x.shape[0]
        out = self.network(x)
        out = out.reshape(n, -1)
        out = self.linear_layer(out)
        return out

    def loss(self, predictions, labels):
        loss_val = self.loss_layer(predictions, labels)
        return loss_val
