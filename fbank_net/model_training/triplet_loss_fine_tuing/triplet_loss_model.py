from .triplet_loss import TripletLoss
from ..base_model import FBankNet


class FBankTripletLossNet(FBankNet):

    def __init__(self, margin):
        super().__init__()
        self.loss_layer = TripletLoss(margin)

    def forward(self, anchor, positive, negative):
        n = anchor.shape[0]
        anchor_out = self.network(anchor)
        anchor_out = anchor_out.reshape(n, -1)
        anchor_out = self.linear_layer(anchor_out)

        positive_out = self.network(positive)
        positive_out = positive_out.reshape(n, -1)
        positive_out = self.linear_layer(positive_out)

        negative_out = self.network(negative)
        negative_out = negative_out.reshape(n, -1)
        negative_out = self.linear_layer(negative_out)

        return anchor_out, positive_out, negative_out

    def loss(self, anchor, positive, negative, reduction='mean'):
        loss_val = self.loss_layer(anchor, positive, negative, reduction)
        return loss_val
