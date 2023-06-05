import torch
from sklearn.metrics import accuracy_score


def goodness_score(pos_acts, neg_acts, threshold=2.0):
    """
    Computes the goodness score for a given set of positive and negative activations.

    Parameters:
      pos_acts (torch.Tensor): Numpy array of positive activations.
      neg_acts (torch.Tensor): Numpy array of negative activations.
      threshold (float, optional): Threshold value used to compute the score. Default is 2.0 .

    Returns:
      goodness (torch.Tensor): Goodness score computed as the sum of positive and negative goodness values. Note that this
      score is actually the quantity that is optimized and not the goodness itself. The goodness itself is the same
      quantity but without the threshold subtraction.
    """

    pos_goodness = -torch.sum(torch.pow(pos_acts, 2)) + threshold
    neg_goodness = torch.sum(torch.pow(neg_acts, 2)) - threshold
    return torch.add(pos_goodness, neg_goodness)


def get_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    return dict(accuracy_score=acc)


