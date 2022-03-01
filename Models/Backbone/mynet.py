import torch
import torch.nn as nn
from typing import Any
import timm


class Net(nn.Module):
    """
    示例：自定义backbone = timm特征提取层 + 自定义分类层
    """

    def __init__(self, pretrained, num_classes):
        super(Net, self).__init__()

        # timm特征提取层
        self.features = timm.create_model("efficientnet_b0", pretrained=pretrained)
        self.features.reset_classifier(0)  # 丢弃timm模型的分类层

        # 自定义分类层
        self.add_linear = nn.Linear(1280, 512, bias=True)
        self.act3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.add_linear(x)
        x = self.act3(x)
        x = self.classifier(x)
        return x


def MyNet(pretrained, num_classes):
    """
    自定义backbone
    """
    return Net(pretrained, num_classes)
