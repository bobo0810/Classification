import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
from pytorch_metric_learning import losses


class create_class_loss(nn.Module):
    """
    常规分类 - 损失函数入口
    """

    def __init__(self, name):
        super(create_class_loss, self).__init__()
        assert name in ["cross_entropy", "label_smooth"]
        self.loss = self.init_loss(name)

    def forward(self, predict, target):
        return self.loss(predict, target)

    def init_loss(self, name):
        """
        常规分类
        """
        loss_dict = {
            "cross_entropy": nn.CrossEntropyLoss,
            "label_smooth": LabelSmoothingCrossEntropy,
        }
        loss = loss_dict[name]()
        return loss


class create_metric_loss(nn.Module):
    """
    度量学习 - 损失函数入口
    """

    def __init__(self, name, num_classes, embedding_size):
        """
        name: 损失函数名称
        num_classes: 类别数
        embedding_size: 特征维度
        """
        super(create_metric_loss, self).__init__()
        assert name in ["cosface", "arcface", "subcenter_arcface", "circleloss"]
        self.loss = self.init_loss(name, num_classes, embedding_size)

    def forward(self, predict, target, hard_tuples):
        return self.loss(predict, target, hard_tuples)

    def init_loss(self, name, num_classes, embedding_size):
        loss_dict = {
            "cosface": losses.CosFaceLoss,
            "arcface": losses.ArcFaceLoss,
            "subcenter_arcface": losses.SubCenterArcFaceLoss,
        }

        if name in loss_dict.keys():
            loss = loss_dict[name](
                num_classes=num_classes, embedding_size=embedding_size
            )
        elif name == "circleloss":
            loss = losses.CircleLoss()
        return loss
