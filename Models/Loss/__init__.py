import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
from pytorch_metric_learning import losses


class create_loss(nn.Module):
    """
    损失函数入口
    """

    def __init__(self, name):
        super(create_loss, self).__init__()

        # 常规分类
        if name in ["cross_entropy", "label_smooth"]:
            self.loss = self.init_class_loss(name)
        # 度量学习
        elif name in ["arcface"]:
            self.loss = self.init_metric_loss(name)
        else:
            raise NotImplementedError

    def forward(self, predict, target):
        return self.loss(predict, target)

    @staticmethod
    def init_class_loss(name):
        """
        常规分类
        """
        # =================常规分类==========================
        if name == "cross_entropy":
            loss = nn.CrossEntropyLoss()
        elif name == "label_smooth":
            loss = LabelSmoothingCrossEntropy()

        return loss

    @staticmethod
    def init_metric_loss(name):
        """
        度量学习
        """
        if name == "arcface":
            loss = losses.SubCenterArcFaceLoss(num_classes=2, embedding_size=128)
            # loss = losses.ArcFaceLoss(num_classes=2, embedding_size=128)

        return loss
