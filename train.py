import sys
import os
import torch
import argparse
from DataSets import create_datasets, create_dataloader
from Utils.eval import eval_model
from Utils.tools import analysis_dataset
from Utils.ddp_tools import init_env, save_model, copy_model, DDP_SummaryWriter
from Models.Backbone import create_backbone
from Models.Loss import create_class_loss
from Models.Optimizer import create_optimizer
from Models.Scheduler import create_scheduler
from torchinfo import summary
import colossalai

cur_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    """分类任务"""
    parser = colossalai.get_default_parser()
    parser.add_argument("--config_file", help="训练配置", default="./Config/config.py")

    # 初始化环境
    ckpt_path, cfg, tb_writer, logger = init_env(parser.parse_args().config_file)

    # 数据集
    dataset = analysis_dataset(cfg.Txt)
    tb_writer.add_dataset_info(dataset)
    train_set = create_datasets(
        dataset=dataset["train"], size=cfg.Size, process=cfg.Process, use_augment=True
    )
    val_set = create_datasets(
        dataset=dataset["val"], size=cfg.Size, process=cfg.Process
    )
    train_dataloader = create_dataloader(cfg.Batch, train_set, cfg.Sampler)
    val_dataloader = create_dataloader(cfg.Batch, val_set)

    # 模型
    model = create_backbone(cfg.Backbone, num_classes=len(dataset["all_labels"]))
    model.info = {"task": "class", "all_labels": dataset["all_labels"]}
    cp_model = copy_model(model)
    tb_writer.add_model_info(model, cfg.Size)

    # 损失函数
    criterion = create_class_loss(cfg.Loss)

    # 优化器
    optimizer = create_optimizer(cfg.Optimizer, model.parameters(), lr=cfg.LR)

    # 学习率调度器
    lr_scheduler = create_scheduler(cfg.Scheduler, cfg.Epochs, optimizer)

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
        tb_writer.add_augment_imgs(epoch, imgs, labels, dataset["all_labels"])
        tb_writer.add_scalar("Train/lr", lr_scheduler.get_last_lr()[0], epoch)
        tb_writer.add_scalar("Val/acc", acc, epoch)
        lr_scheduler.step()
    save_model(engine.model, cp_model, ckpt_path + cfg.Backbone + "_last.pt")
    tb_writer.close()
