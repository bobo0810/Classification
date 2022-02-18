import torch
import torch.nn as nn
from .sam import SAM


def Optimizer(model, name, lr):
    """
    优化器入口
    """
    if name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
    elif name == "SAM":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            model.parameters(), base_optimizer, rho=0.05, lr=lr, weight_decay=0.0005
        )
    else:
        raise NotImplementedError
    return optimizer
