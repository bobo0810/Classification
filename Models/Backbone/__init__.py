import torch.nn as nn
import timm
import torch
from collections import OrderedDict
from .mynet import mynet  # 注册自定义分类模型
from .mynet_metric import mynet_metric  # 注册自定义特征提取模型


def create_backbone(model_name, num_classes, metric=False):
    """
    主干网络入口

    model_name:  timm模型名称
    num_classes: 分类时为类别数   度量学习时为特征维度
    metric:      False分类任务   True度量学习
    """
    if metric:
        model = MetricModel(model_name, pretrained=True, feature_dim=num_classes)
        model.metric = True  # 区分任务的标志位
        return model
    else:
        return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


class MetricModel(nn.Module):
    """
    度量学习：加载特征提取网络
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
