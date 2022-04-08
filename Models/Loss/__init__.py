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
            self.task = "class"
        # 度量学习
        elif name in ["cosface", "arcface", "subcenter_arcface", "circleloss"]:
            self.loss = self.init_metric_loss(name)
            self.task = "metric"
        else:
            raise NotImplementedError

    def forward(self, predict, target):
        return self.loss(predict, target)

    def init_class_loss(self, name):
        """
        常规分类
        """
        loss_dict = {
            "cross_entropy": nn.CrossEntropyLoss,
            "label_smooth": LabelSmoothingCrossEntropy,
        }
        loss = loss_dict[name]()
        return loss

    def init_metric_loss(self, name, num_classes=2, embedding_size=128):
        """
        度量学习
        num_classes: 类别数
        embedding_size: backbone输出的特征维度
        """
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
