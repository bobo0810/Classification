import torch
import torch.nn as nn


def create_optimizer(model, name, lr):
    """
    优化器入口
    """
    if name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
    else:
        raise NotImplementedError
    return optimizer
