import torch
import torch.nn as nn
from .focal_loss import FocalLoss


class create_loss(nn.Module):
    """
    损失函数入口
    """

    def __init__(self, name):
        super(create_loss, self).__init__()
        self.loss = self.init_loss(name)

    def forward(self, predict, target):
        return self.loss(predict, target)

    @staticmethod
    def init_loss(name):
        """
        初始化损失函数
        """
        if name == "cross_entropy":
            loss = nn.CrossEntropyLoss()
        elif name == "focal_loss":
            loss = FocalLoss()
        else:
            raise NotImplementedError
        return loss
