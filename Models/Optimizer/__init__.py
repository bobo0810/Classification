import timm
import timm.optim


def SGD(params, lr):
    """SGD优化器"""
    return timm.optim.create_optimizer_v2(
        params, opt="sgd", lr=lr, weight_decay=0.0005, momentum=0.9
    )


def Adam(params, lr):
    """Adam优化器"""
    return timm.optim.create_optimizer_v2(
        params, opt="adam", lr=lr, weight_decay=0.0005, momentum=0.9
    )


def Lamb(params, lr):
    """Lamb优化器"""
    return timm.optim.create_optimizer_v2(
        params, opt="lamb", lr=lr, weight_decay=0.0005, momentum=0.9
    )


# ======================================加载==========================================
def create_optimizer(opt_name, params, lr):
    """
    优化器入口

    opt_name: 优化器名称
    params: 模型或参数
    lr: 初始学习率
    """
    try:
        method = eval(opt_name)
        return method(params, lr)
    except:
        raise NotImplemented
