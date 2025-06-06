import numpy as np
import torch


# -- mixup data augmentation
# from https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
def mixup_data(x, y, alpha=1.0, soft_labels=None, use_cuda=False):
    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda"""

    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(torch.device("cpu"))
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(
        pred, y_b
    )
