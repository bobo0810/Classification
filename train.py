import sys
import os
import torch
import argparse
from DataSets import create_datasets, create_dataloader
from Utils.tools import analysis_dataset, eval_model
from Utils.ddp_tools import init_env, save_model, copy_model, DDP_SummaryWriter
from Models.Backbone import create_backbone
from Models.Loss import create_class_loss
from Models.Optimizer import create_optimizer
from Models.Scheduler import create_scheduler
from torchinfo import summary
import colossalai

cur_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    '''分类任务'''
    parser = colossalai.get_default_parser()
    parser.add_argument("--config_file", help="训练配置", default="./Config/config.py")

    # 初始化环境
    ckpt_path, tb_path, cfg, logger = init_env(parser.parse_args().config_file)

    # 模型
    labels_list = analysis_dataset(cfg.Txt)["labels"]
    model = create_backbone(cfg.Backbone, num_classes=len(labels_list))
    model.info = {"task": "class", "labels": labels_list}  # 额外信息
    cp_model = copy_model(model)

    # 损失函数
    criterion = create_class_loss(cfg.Loss)

    # 优化器
    optimizer = create_optimizer( cfg.Optimizer, model.parameters(),lr=cfg.LR)

    # 学习率调度器
    lr_scheduler = create_scheduler(cfg.Scheduler, cfg.Epochs, optimizer)

    # 数据集
    train_set = create_datasets(
        txt=cfg.Txt, mode="train", size=cfg.Size, process=cfg.Process, use_augment=True
    )
    val_set = create_datasets(
        txt=cfg.Txt, mode="val", size=cfg.Size, process=cfg.Process
    )

    # 数据集加载器
    train_dataloader = create_dataloader(cfg.Batch, train_set, cfg.Sampler)
    val_dataloader = create_dataloader(cfg.Batch, val_set)

    # 日志
    logger.info(f"tensorboard save in {tb_path}", ranks=[0])
    tb_writer = DDP_SummaryWriter(tb_path)

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

        # 验证集评估
        engine.eval()
        acc = eval_model(engine, val_dataloader).Overall_ACC
        if best_acc <= acc:
            best_acc = acc
            save_model(engine.model, cp_model, ckpt_path + cfg.Backbone + "_best.pt")

        # 可视化
        tb_writer.add_augment_imgs(epoch, imgs, labels, labels_list)
        tb_writer.add_scalar("Train/lr", lr_scheduler.get_last_lr()[0], epoch)
        tb_writer.add_scalar("Eval/acc", acc, epoch)
        lr_scheduler.step()
    save_model(engine.model, cp_model, ckpt_path + cfg.Backbone + "_last.pt")
    tb_writer.close()
