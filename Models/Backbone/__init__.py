import torch.nn as nn
import torch.nn.functional as F
import timm
import torch
from collections import OrderedDict
from .backbone import create_custom_backbone  # 导入自定义网络


def create_backbone(model_name, num_classes, metric=False):
    """
    主干网络入口
    优先顺序: 自定义>timm

    model_name:  网络名称
    num_classes: 网络输出
    metric:      False分类任务   True度量学习
    """
    try:
        # 加载自定义网络
        model = create_custom_backbone(model_name, num_classes)
    except:
        # 加载Timm网络
        if metric:
            model = MetricModel(model_name, pretrained=True, feature_dim=num_classes)
        else:
            model = timm.create_model(
                model_name, pretrained=True, num_classes=num_classes
            )
    return model


class MetricModel(nn.Module):
    """
    度量学习：加载基于timm的特征提取网络
    """

    def __init__(self, model_name, pretrained, feature_dim):
        super(MetricModel, self).__init__()
        # 特征提取器
        self.features = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=feature_dim,  # 修改输出维度
        )
        self.bn = nn.BatchNorm1d(feature_dim)

    def forward(self, imgs):
        features = self.features(imgs)
        features = self.bn(features)  # 规范化，正则化
        features = F.normalize(features, p=2, dim=1)  # 特征归一化，即模长为1
        return features
