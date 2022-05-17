import sys
import os
import torch
import argparse
from DataSets import create_datasets, create_dataloader
from Utils.tools import analysis_dataset, init_env, eval_metric_model
from Utils.ddp_tsdb import DDP_SummaryWriter
from Utils.ddp_tools import create_folder, save_model, copy_model
from Models.Backbone import create_backbone
from Models.Loss import create_metric_loss
from Models.Optimizer import create_optimizer
from torchinfo import summary
from pytorch_metric_learning import miners
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
    model = create_backbone(cfg.Backbone, num_classes=cfg.Feature_dim)
    model.metric = True  # 区分任务的标志位
    cp_model = copy_model(model, cur_rank)

    # 分类器
    mining_func = miners.MultiSimilarityMiner()  # 难样例挖掘
    criterion = create_metric_loss(cfg.Loss, cfg.Feature_dim, len(labels_list))

    # 优化器
    params = [
        {"params": model.parameters(), "lr": cfg.LR},
        {"params": criterion.parameters(), "lr": cfg.LR},
    ]
    optimizer = create_optimizer(params, cfg.Optimizer, lr=cfg.LR)

    # 学习率调度器
    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer, cfg.Epochs, warmup_steps=int(cfg.Epochs * 0.1)
    )

    # 数据集
    train_set_for_val = create_datasets(
        txt=cfg.Txt, mode="train", size=cfg.Size
    )  # 用于评估
    train_set = create_datasets(
        txt=cfg.Txt, mode="train", size=cfg.Size, use_augment=True
    )  # 用于训练
    val_set = create_datasets(txt=cfg.Txt, mode="val", size=cfg.Size)

    # 数据集加载器
    train_dataloader = create_dataloader(cfg.Batch, train_set, cfg.Sampler)

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

        ##############
        # 验证集评估  #
        ##############
        engine.eval()
        precision = eval_metric_model(engine, train_set_for_val, val_set)
        if best_score <= precision:
            best_score = precision
            save_model(
                engine.model, cp_model, ckpt_path + cfg.Backbone + "_best.pt", cur_rank
            )

        # 可视化
        tb_writer.add_augment_imgs(epoch, imgs, labels, labels_list)
        tb_writer.add_scalar("Train/lr", lr_scheduler.get_last_lr()[0], epoch)
        tb_writer.add_scalar("Eval/precision", precision, epoch)
        lr_scheduler.step()
    save_model(engine.model, cp_model, ckpt_path + cfg.Backbone + "_last.pt", cur_rank)
    tb_writer.close()
