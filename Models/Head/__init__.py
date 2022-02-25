import torch
import torch.nn as nn
from .focal_loss import FocalLoss


class create_head(nn.Module):
    """
    损失函数入口
    """

    def __init__(self, name):
        super(create_head, self).__init__()
        self.head = self.init_head(name)

    def forward(self, predict, target):
        return self.head(predict, target)

    @staticmethod
    def init_head(name):
        """
        初始化损失函数
        """
        if name == "cross_entropy":
            head = nn.CrossEntropyLoss()
        elif name == "focal_loss":
            head = FocalLoss()
        else:
            raise NotImplementedError
        return head
