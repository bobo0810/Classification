from .optimizer import *  # 导入优化器


def create_optimizer(opt_name, params, lr):
    """
    优化器入口

    opt_name: 优化器名称
    params: 模型或参数
    lr: 初始学习率
    """
    try:
        return eval(opt_name)(params, lr)
    except:
        raise NotImplemented
