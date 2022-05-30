import sys
import os
import torch
import argparse
from DataSets import create_datasets, create_dataloader
from Utils.tools import analysis_dataset, eval_metric_model
from Utils.ddp_tools import init_env,save_model, copy_model, DDP_SummaryWriter
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
    parser = colossalai.get_default_parser()
    parser.add_argument("--config_file", help="训练配置", default="./Config/config.py")

    # 初始化环境
    colossalai.launch_from_torch(config=parser.parse_args().config_file)
    cfg = gpc.config
    logger = get_dist_logger()
    ckpt_path, tb_path = init_env()


    # 模型
    labels_list = analysis_dataset(cfg.Txt)["labels"]
    model = create_backbone(cfg.Backbone, cfg.Feature_dim, metric=True)
    model.info = {"task": "metric", "labels": labels_list}  # 额外信息
    cp_model = copy_model(model)

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
    lr_scheduler = create_scheduler(cfg.Scheduler, cfg.Epochs, optimizer)

    # 数据集
    train_set_for_val = create_datasets(
        txt=cfg.Txt, mode="train", size=cfg.Size, process=cfg.Process
    )  # 用于评估
    train_set = create_datasets(
        txt=cfg.Txt,
        mode="train",
        size=cfg.Size,
        process=cfg.Process,
        use_augment=True,
    )  # 用于训练
    val_set = create_datasets(txt=cfg.Txt, mode="val", size=cfg.Size,process=cfg.Process)

    # 数据集加载器
    train_dataloader = create_dataloader(cfg.Batch, train_set, cfg.Sampler)

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
        precision = eval_metric_model(engine, train_set_for_val, val_set, cfg.Batch)
        if best_score <= precision:
            best_score = precision
            save_model(engine.model, cp_model, ckpt_path + cfg.Backbone + "_best.pt")

        # 可视化
        tb_writer.add_augment_imgs(epoch, imgs, labels, labels_list)
        tb_writer.add_scalar("Train/lr", lr_scheduler.get_last_lr()[0], epoch)
        tb_writer.add_scalar("Eval/precision", precision, epoch)
        lr_scheduler.step()
    save_model(engine.model, cp_model, ckpt_path + cfg.Backbone + "_last.pt")
    tb_writer.close()
