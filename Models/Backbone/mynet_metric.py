import torch
import torch.nn as nn
from typing import Any
import timm
from timm.models import register_model
from pytorch_metric_learning import losses


def l2_norm(x, axis=1):
    norm = torch.norm(x, 2, axis, True)
    output = x / norm
    return output


class MyNet_Metric(nn.Module):
    """
    特征提取网络

    训练时 特征提取器->分类器，输出loss
    推理时 特征提取器 输出feature
    """

    def __init__(self, pretrained, num_classes, model_name, embedding_size):
        super(MyNet_Metric, self).__init__()
        self.task = "metric"  # (!!!标志位 必须保留!!!)区分 常规分类or度量学习

        # 特征提取器
        self.features = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=embedding_size,  # 修改输出维度
        )
        self.bn = nn.BatchNorm1d(embedding_size)
        # 分类器
        self.classifier = losses.SubCenterArcFaceLoss(
            num_classes=num_classes, embedding_size=embedding_size
        )

    def forward(self, imgs, labels):
        features = self.features(imgs)
        features = self.bn(features)
        features = l2_norm(features)

        loss = self.classifier(features, labels)
        return loss


"""
注意:
1. @register_model注册为timm模型
2. 命名尽量避免与timm模型重名
"""


@register_model
def mynet_metric(
    pretrained, num_classes, model_name="efficientnet_b0", embedding_size=128
):
    """
    pretrained: 是否加载ImageNet预训练参数
    num_classes: 类别数
    model_name: timm主干网络名
    embedding_size: timm主干网络输出的特征维度
    """
    print("Backbone_Metric come from user-defined")
    model = MyNet_Metric(pretrained, num_classes, model_name, embedding_size)
    return model
