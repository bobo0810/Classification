import sys
import os
import torch
import argparse
import copy
from DataSets.preprocess import PreProcess
from DataSets import create_datasets, create_dataloader
from Utils.tools import analysis_dataset, init_env, eval_model, eval_metric_model
from Models.Backbone import create_backbone
from Models.Loss import create_class_loss, create_metric_loss
from Models.Optimizer import create_optimizer
from Utils.tools import tensor2img
from torchinfo import summary
from colossalai.core import global_context as gpc 
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.logging import get_dist_logger
from torch.utils.tensorboard import SummaryWriter
import colossalai
cur_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    parser = colossalai.get_default_parser()
    parser.add_argument("--config_file",help="训练配置", default="./Config/config.py")

    # 初始化环境
    colossalai.launch_from_torch(config=parser.parse_args().config_file) 
    cfg=gpc.config
    cur_rank=gpc.get_global_rank()
    logger = get_dist_logger()
    exp_path = init_env()
 
    # 模型
    labels_list = analysis_dataset(cfg.Txt)["labels"]
    model = create_backbone(cfg.Backbone, num_classes=len(labels_list))
    
    # 数据集
    train_set = create_datasets(txt=cfg.Txt, mode="train", size=cfg.Size, use_augment=True,)
    val_set = create_datasets(txt=cfg.Txt, mode="val", size=cfg.Size)
    
    # 数据集加载器
    train_dataloader = create_dataloader(cfg.Batch,train_set,cfg.Sampler)
    val_dataloader = create_dataloader(cfg.Batch, val_set)

    # 损失函数
    criterion = create_class_loss(cfg.Loss)
    params = []


    # 优化器
    params.append({"params": model.parameters()})
    optimizer = create_optimizer(params, cfg.Optimizer, lr=cfg.LR)

    # 学习率调度器
    lr_scheduler = CosineAnnealingWarmupLR(optimizer, total_steps=cfg.Epochs,warmup_steps=int(cfg.Epochs*0.1))
    
    # 日志
    if cur_rank==0:
        # 初始化
        os.makedirs(os.path.join(exp_path,"checkpoint/"))
        tb_writer = SummaryWriter(os.path.join(exp_path,"tb_log/"))
        logger.info(f"Log in {exp_path}")

        # 参数
        tb_writer.add_text("Config", str(cfg))

        # 数据集
        tb_writer.add_text("TrainSet", train_set.get_info())
        tb_writer.add_text("ValSet", val_set.get_info())
        tb_writer.close()
    
    # colossalai封装
    engine, train_dataloader, val_dataloader, _ = colossalai.initialize(
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader,
    )

    best_score = 0.0
    for epoch in range(cfg.Epochs):
        engine.train()
        logger.info(f"Starting {epoch} / {cfg.Epochs}",ranks=[0])
        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            imgs, labels = imgs.cuda(), labels.cuda()
            engine.zero_grad()
            output = engine(imgs)
            loss = engine.criterion(output, labels)
            engine.backward(loss)
            engine.step()

            if batch_idx % 100 == 0 and cur_rank == 0:
                iter_num = int(batch_idx + epoch * len(train_dataloader))
                tb_writer.add_scalar("Train/loss", loss.item(), iter_num)
        lr_scheduler.step()

        # 验证集评估
        engine.eval()
        ##############
        #pass
        ##############
        if cur_rank == 0:
            tb_writer.add_scalar("Train/lr", lr_scheduler.get_last_lr()[0], epoch)

    if cur_rank == 0:
        tb_writer.close()

# 运行
# colossalai run --nproc_per_node 2 train.py