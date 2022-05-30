from .scheduler import *  # 导入调度器


def create_scheduler(sched_name, epochs, optimizer):
    """
    学习率调度器入口

    sched_name: 调度器名称
    epochs: 总轮数
    optimizer: 优化器
    """
    try:
        return eval(sched_name)(epochs, optimizer)
    except:
        raise NotImplemented
