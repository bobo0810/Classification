from timm.scheduler import create_scheduler as timm_create_scheduler

scheduler_list = [
    "multistep",
    "cosine",
]


class Params:
    """
    默认参数
    """

    def __init__(self, epochs, init_lr, sched):
        self.epochs = epochs  # 总轮数
        self.sched = sched  # 调度器名称
        self.warmup_lr = init_lr / 2  # 预热学习率
        self.warmup_epochs = epochs * 0.1  # 预热轮数
        self.cooldown_epochs = 0
        self.min_lr = 1e-5  # 学习率下限


def create_scheduler(sched_name, epochs, optimizer):
    """
    学习率调度器入口

    sched_name: 调度器名称
    epochs: 总轮数
    lr: 学习率
    optimizer: 优化器
    """
    assert sched_name in scheduler_list, "NotImplementedError"
    lr = optimizer.param_groups[0]["lr"]

    if sched_name == "multistep":
        # 配置参数
        params = Params(epochs, lr, sched_name)
        params.decay_epochs = [int(epochs * 0.7), int(epochs * 0.9)]
        params.decay_rate = 0.1

    elif sched_name == "cosine":
        # 配置参数
        params = Params(epochs, lr, sched_name)

    else:
        raise NotImplementedError
    scheduler, _ = timm_create_scheduler(params, optimizer)
    return scheduler
