import torch
import torch.nn as nn
from typing import Any
import timm
from timm.models import register_model


class MyNet(nn.Module):
    """
    示例: 自定义backbone = timm特征提取层 + 自定义分类层
    """

    def __init__(self, pretrained, num_classes):
        super(MyNet, self).__init__()

        # timm特征提取层,丢弃分类层
        self.features = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )

        # 自定义分类层
        self.add_linear = nn.Linear(1280, 512, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.add_linear(x)
        x = self.act(x)
        x = self.classifier(x)
        return x


"""
注意:
1. @register_model注册为timm模型
2. 命名尽量避免与timm模型重名
"""


@register_model
def mynet(pretrained, num_classes):
    print("Backbone come from user-defined")
    model = MyNet(pretrained, num_classes)
    return model
