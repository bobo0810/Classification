from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR

scheduler_list = [
    "cosine",
]


def create_scheduler(sched_name, epochs, optimizer):
    """
    学习率调度器入口

    sched_name: 调度器名称
    epochs: 总轮数
    optimizer: 优化器
    """
    if sched_name == "cosine":
        # warmup_steps 预热轮数 
        lr_scheduler = CosineAnnealingWarmupLR(
            optimizer, epochs, warmup_steps=int(epochs * 0.1)  
        )
    else:
        raise NotImplementedError()
    return lr_scheduler
