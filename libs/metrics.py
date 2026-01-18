import torch.nn.functional


def active_samples(emb, labels, pos_margin, neg_margin):
    sim = emb @ emb.T

    if labels.ndim == 1:
        labels = labels[:, None]

    pos_mask = labels == labels.T
    neg_mask = labels != labels.T

    # active positive samples = loss smaller than pos_margin:
    active_pos = torch.logical_and(torch.greater(pos_margin * torch.ones_like(sim), sim), pos_mask).to(dtype=torch.float32).sum()

    # active negative samples = loss greater than neg_margin:
    active_neg = torch.logical_and(torch.greater(sim, neg_margin * torch.ones_like(sim)), neg_mask).to(dtype=torch.float32).sum()

    return active_pos, active_neg