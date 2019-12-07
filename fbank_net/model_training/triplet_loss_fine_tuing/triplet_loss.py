import torch
from torch import nn


class TripletLoss(nn.Module):

    def __init__(self, margin):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity()
        self.margin = margin

    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings, reduction='mean'):

        # cosine distance is a measure of dissimilarity. The higher the value, more the two vectors are dissimilar
        # it is calculated as (1 - cosine similarity) and ranges between (0,2)

        positive_distance = 1 - self.cosine_similarity(anchor_embeddings, positive_embeddings)
        negative_distance = 1 - self.cosine_similarity(anchor_embeddings, negative_embeddings)

        losses = torch.max(positive_distance - negative_distance + self.margin,torch.full_like(positive_distance, 0))
        if reduction == 'mean':
            return torch.mean(losses)
        else:
            return torch.sum(losses)
