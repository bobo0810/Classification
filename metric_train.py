import sys
import os
import torch
import argparse
from DataSets import create_datasets, create_dataloader
from Utils.eval import eval_metric_model
from Utils.tools import analysis_dataset
from Utils.ddp_tools import init_env, save_model, copy_model
from Models.Backbone import create_backbone
from Models.Loss import create_metric_loss
from Models.Optimizer import create_optimizer
from Models.Scheduler import create_scheduler
from torchinfo import summary
from pytorch_metric_learning import miners
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
import colossalai

cur_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    """度量学习"""
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
    train_dataloader = create_dataloader(cfg.Batch, train_set, cfg.Sampler)

    # 模型
    model = create_backbone(cfg.Backbone, cfg.Feature_dim, metric=True)
    model.info = {"task": "metric", "all_labels": dataset["all_labels"]}  # 额外信息
    cp_model = copy_model(model)
    tb_writer.add_model_info(model, cfg.Size)

    # 损失函数/分类器
    mining_func = miners.MultiSimilarityMiner()  # 难样例挖掘
    criterion = create_metric_loss(
        cfg.Loss, cfg.Feature_dim, len(dataset["all_labels"])
    )

    # 优化器
    params = [
        {"params": model.parameters(), "lr": cfg.LR},
        {"params": criterion.parameters(), "lr": cfg.LR},
    ]
    optimizer = create_optimizer(cfg.Optimizer, params, lr=cfg.LR)

    # 学习率调度器
    lr_scheduler = create_scheduler(cfg.Scheduler, cfg.Epochs, optimizer)

    # colossalai封装
    engine, train_dataloader, _, _ = colossalai.initialize(
        model,
        optimizer,
        criterion,
        train_dataloader,
    )

    best_score = 0.0
    for epoch in range(cfg.Epochs):
        engine.train()
        logger.info(f"Starting {epoch} / {cfg.Epochs}", ranks=[0])
        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            imgs, labels = imgs.cuda(), labels.cuda()
            engine.zero_grad()
            output = engine(imgs)
            hard_tuples = mining_func(output, labels)
            loss = engine.criterion(output, labels, hard_tuples)
            engine.backward(loss)
            engine.step()

            if batch_idx % 100 == 0:
                iter_num = int(batch_idx + epoch * len(train_dataloader))
                tb_writer.add_scalar("Train/loss", loss.item(), iter_num)

        # 验证集评估
        engine.eval()
        score = eval_metric_model(
            engine, dataset, cfg.Size, cfg.Process, cfg.Batch, mode="val"
        )
        if best_score <= score["value"]:
            best_score = score["value"]
            save_model(engine.model, cp_model, ckpt_path + cfg.Backbone + "_best.pt")

        # 可视化
        tb_writer.add_augment_imgs(epoch, imgs, labels, dataset["all_labels"])
        tb_writer.add_scalar("Train/lr", lr_scheduler.get_last_lr()[0], epoch)
        tb_writer.add_scalar("Val/" + score["index"], score["value"], epoch)
        lr_scheduler.step()
    save_model(engine.model, cp_model, ckpt_path + cfg.Backbone + "_last.pt")
    tb_writer.close()
