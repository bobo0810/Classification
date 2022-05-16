import sys
import os
import torch
import argparse
from DataSets import create_datasets, create_dataloader
from Utils.tools import analysis_dataset, init_env, eval_model
from Utils.ddp_tsdb import DDP_SummaryWriter
from Utils.ddp_tools import create_folder, save_model, copy_model
from Models.Backbone import create_backbone
from Models.Loss import create_class_loss
from Models.Optimizer import create_optimizer
from torchinfo import summary
from colossalai.core import global_context as gpc
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.logging import get_dist_logger
import colossalai

cur_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    parser = colossalai.get_default_parser()
    parser.add_argument("--config_file", help="训练配置", default="./Config/config.py")

    # 初始化环境
    colossalai.launch_from_torch(config=parser.parse_args().config_file)
    cfg = gpc.config
    cur_rank = gpc.get_global_rank()
    logger = get_dist_logger()
    ckpt_path, tb_path = init_env()
    create_folder(ckpt_path, cur_rank)

    # 模型
    labels_list = analysis_dataset(cfg.Txt)["labels"]
    model = create_backbone(cfg.Backbone, num_classes=len(labels_list))
    cp_model = copy_model(model, cur_rank)

    # 损失函数
    criterion = create_class_loss(cfg.Loss)

    # 优化器
    optimizer = create_optimizer(model.parameters(), cfg.Optimizer, lr=cfg.LR)

    # 学习率调度器
    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer, cfg.Epochs, warmup_steps=int(cfg.Epochs * 0.1)
    )

    # 数据集
    train_set = create_datasets(
        txt=cfg.Txt, mode="train", size=cfg.Size, use_augment=True
    )
    val_set = create_datasets(txt=cfg.Txt, mode="val", size=cfg.Size)

    # 数据集加载器
    train_dataloader = create_dataloader(cfg.Batch, train_set, cfg.Sampler)
    val_dataloader = create_dataloader(cfg.Batch, val_set)

    # 日志
    logger.info(f"tensorboard save in {tb_path}", ranks=[0])
    tb_writer = DDP_SummaryWriter(tb_path, rank=cur_rank)

    # 参数可视化
    tb_writer.add_text("Config", str(cfg))

    # 数据集可视化
    tb_writer.add_text("TrainSet", train_set.get_info())
    tb_writer.add_text("ValSet", val_set.get_info())

    # 模型结构可视化
    tb_writer.add_graph(model, cfg.Size)

    # colossalai封装
    engine, train_dataloader, val_dataloader, _ = colossalai.initialize(
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader,
    )

    best_acc = 0.0
    for epoch in range(cfg.Epochs):
        engine.train()
        logger.info(f"Starting {epoch} / {cfg.Epochs}", ranks=[0])
        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            imgs, labels = imgs.cuda(), labels.cuda()
            engine.zero_grad()
            output = engine(imgs)
            loss = engine.criterion(output, labels)
            engine.backward(loss)
            engine.step()

            if batch_idx % 100 == 0:
                iter_num = int(batch_idx + epoch * len(train_dataloader))
                tb_writer.add_scalar("Train/loss", loss.item(), iter_num)

        ##############
        # 验证集评估  #
        ##############
        engine.eval()
        acc = eval_model(engine, val_dataloader).Overall_ACC
        if best_acc <= acc:
            best_acc = acc
            save_model(
                engine.model, cp_model, ckpt_path + cfg.Backbone + "_best.pt", cur_rank
            )

        # 可视化
        tb_writer.add_augment_imgs(epoch, imgs, labels, labels_list)
        tb_writer.add_scalar("Train/lr", lr_scheduler.get_last_lr()[0], epoch)
        tb_writer.add_scalar("Eval/acc", acc, epoch)
        lr_scheduler.step()
    save_model(engine.model, cp_model, ckpt_path + cfg.Backbone + "_last.pt", cur_rank)
    tb_writer.close()

# 运行
# colossalai run --nproc_per_node 2 train.py
