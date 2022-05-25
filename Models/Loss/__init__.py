import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
from pytorch_metric_learning import losses

# 当前支持
classloss_list = ["cross_entropy", "label_smooth"]
metricloss_list = ["cosface", "arcface", "subcenter_arcface", "circleloss"]


class create_class_loss(nn.Module):
    """
    常规分类 - 损失函数入口
    """

    def __init__(self, name):
        super(create_class_loss, self).__init__()
        assert name in classloss_list, "NotImplementedError"
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

    def __init__(self, name, feature_dim, num_classes):
        """
        name: 损失函数名称
        feature_dim: 特征维度
        num_classes: 类别数
        """
        super(create_metric_loss, self).__init__()
        assert name in metricloss_list, "NotImplementedError"
        self.loss = self.init_loss(name, num_classes, feature_dim)

    def forward(self, predict, target, hard_tuples):
        return self.loss(predict, target, hard_tuples)

    def init_loss(self, name, num_classes, feature_dim):
        loss_dict = {
            "cosface": losses.CosFaceLoss,
            "arcface": losses.ArcFaceLoss,
            "subcenter_arcface": losses.SubCenterArcFaceLoss,
        }

        if name in loss_dict.keys():
            loss = loss_dict[name](num_classes=num_classes, embedding_size=feature_dim)
        elif name == "circleloss":
            loss = losses.CircleLoss()
        return loss
