from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR, MultiStepWarmupLR

# 当前支持
scheduler_list = ["cosine", "multistep"]


def create_scheduler(sched_name, epochs, optimizer):
    """
    学习率调度器入口

    sched_name: 调度器名称
    epochs: 总轮数
    optimizer: 优化器
    """
    assert sched_name in scheduler_list, "NotImplementedError"
    if sched_name == "cosine":
        scheduler = CosineAnnealingWarmupLR(
            optimizer, epochs, warmup_steps=int(epochs * 0.1)
        )
    elif sched_name == "multistep":
        scheduler = MultiStepWarmupLR(
            optimizer,
            epochs,
            warmup_steps=int(epochs * 0.1),  # 预热轮数
            milestones=[int(epochs * 0.7), int(epochs * 0.9)],  # 总轮数的70%、90%时调整学习率
            gamma=0.1,  # 学习率下降的倍数
        )
    else:
        raise NotImplementedError()
    return scheduler
