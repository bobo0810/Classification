from .loss import *  # 导入损失函数


def create_class_loss(name):
    """
    分类学习-损失函数入口
    """
    try:
        return eval(name)()
    except:
        raise NotImplemented


def create_metric_loss(name, feature_dim, num_classes):
    """
    度量学习 - 损失函数入口

    name: 损失函数名称
    feature_dim: 特征维度
    num_classes: 类别数
    """
    try:
        return eval(name)(feature_dim, num_classes)
    except:
        raise NotImplemented
