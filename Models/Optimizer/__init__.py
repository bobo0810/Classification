import torch
import torch.nn as nn


def create_optimizer(model, name, lr):
    """
    优化器入口
    """
    if name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
    elif name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError
    return optimizer
