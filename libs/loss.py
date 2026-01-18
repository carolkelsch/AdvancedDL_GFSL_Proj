
import torch

class ContrastiveLossV2:
    """
    Variant of the contrastive loss using cosine similarity instead of the Euclidean distance, and two margin for
    matching and mismatching samples. Note that this loss requires that the embedding space is L2-normalised. Also,
    the cosine similarity is not a distance, so instead of minimizing it, we have to maximize it for matching samples.

    If vectors have a norm of 1, then the cosine similarity can only vary from -1 to +1. We must have
    `pos_margin` > `neg_margin`. Practically, a positive margin above 0.6 and a negative margin below 0.4 are usual
    values. Not that we don't want the margin to be negative (albeit the cosine similarity might), because it would make
    the embedding space impossible de learn.
    """
    def __init__(self, pos_margin=0.8, neg_margin=0.3):
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin

    def __call__(self, x: torch.Tensor, labels: torch.Tensor, **kwargs):
        x = torch.nn.functional.normalize(x, dim=1)
        sim = x @ x.T

        if labels.ndim == 1:
            labels = labels[:, None]

        pos_mask = labels == labels.T
        neg_mask = labels != labels.T

        pos_loss = torch.sum(pos_mask * torch.relu(self.pos_margin - sim)) / torch.sum(pos_mask)
        neg_loss = torch.sum(torch.relu(sim - self.neg_margin) * neg_mask) / torch.sum(neg_mask)

        return pos_loss + neg_loss
