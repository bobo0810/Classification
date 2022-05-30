import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import timm

# ClassNet     分类网络示例
# MetricNet    特征提取网络示例

class ClassNet(nn.Module):
    """
    示例: 自定义backbone = timm特征提取层 + 自定义分类层
    """

    def __init__(self,num_classes):
        super(ClassNet, self).__init__()

        # timm特征提取层,丢弃分类层
        self.features = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
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


class MetricNet(nn.Module):
    """
    示例: 特征提取网络 输出feature
    """

    def __init__(self, num_classes):
        super(MetricNet, self).__init__()
        feature_dim = num_classes
        
        # 特征提取器
        self.features = timm.create_model(
            model_name="efficientnet_b0",
            pretrained=True,
            num_classes=feature_dim,  # 修改输出维度
        )
        self.bn = nn.BatchNorm1d(feature_dim)

    def forward(self, imgs):
        features = self.features(imgs)
        features = self.bn(features)  # 规范化，正则化
        features = F.normalize(features, p=2, dim=1)  # 特征归一化，即模长为1
        return features
