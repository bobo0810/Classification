import sys
import os
import torch
import argparse
import copy
from DataSets.preprocess import PreProcess
from DataSets import create_datasets, create_dataloader
from Utils.tools import analysis_dataset, init_env, eval_model, eval_metric_model,object2dict
from Models.Backbone import create_backbone
from Models.Loss import create_class_loss, create_metric_loss
from Models.Optimizer import create_optimizer
from Models.Scheduler import create_scheduler
from Utils.tools import tensor2img
from timm.utils import ModelEmaV2
from torchinfo import summary
from pytorch_metric_learning import miners

cur_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":

    # !导入训练配置!
    from Config.config import cfg

    # 初始化环境
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_list = analysis_dataset(cfg.txt)["labels"]

    tb_writer, ckpt_path = init_env()
    tb_writer.add_text("Config", str(object2dict(cfg)))
    ckpt_path+= cfg.backbone

    # 模型
    model = create_backbone(cfg.backbone, num_classes=len(labels_list))
    vis_model = copy.deepcopy(model)
    TASK = "metric" if hasattr(model, "embedding_size") else "class"
    # 度量学习
    if TASK == "metric":
        # 数据集
        train_set = create_datasets(txt=cfg.txt, mode="train", size=cfg.size, use_augment=True)
        train_dataloader = create_dataloader(batch_size=cfg.batch, dataset=train_set, sampler_name=cfg.sampler)
        val_set = create_datasets(txt=cfg.txt, mode="val", size=cfg.size)

        # 难样例挖掘
        mining_func = miners.MultiSimilarityMiner()

        # 损失函数(分类器)
        loss_func = create_metric_loss(
            name=cfg.loss,
            num_classes=len(labels_list),
            embedding_size=model.embedding_size,
        ).to(device)
        params = [{"params": loss_func.parameters(), "lr": cfg.lr}]
    # 常规分类
    else:
        # 数据集
        train_set = create_datasets(txt=cfg.txt, mode="train", size=cfg.size, use_augment=True,)
        val_set = create_datasets(txt=cfg.txt, mode="val", size=cfg.size)

        # 数据集加载器
        train_dataloader = create_dataloader(
            batch_size=cfg.batch,
            dataset=train_set,
            sampler_name=cfg.sampler,
        )

        val_dataloader = create_dataloader(batch_size=cfg.batch, dataset=val_set)

        # 损失函数
        loss_func = create_class_loss(cfg.loss).to(device)
        params = []

    # 模型转为GPU
    if device != "cpu":
        model = torch.nn.DataParallel(model).to(device)
    model.train()
    ema_model = ModelEmaV2(model, decay=0.9998)

    # 优化器
    params.append({"params": model.parameters()})
    optimizer = create_optimizer(params, cfg.optimizer, lr=cfg.lr)

    # 学习率调度器
    lr_scheduler = create_scheduler(
        sched_name=cfg.scheduler,
        epochs=cfg.epochs,
        optimizer=optimizer,
    )
    best_score = 0.0
    for epoch in range(cfg.epochs):
        print("start epoch {}/{}...".format(epoch, cfg.epochs))
        tb_writer.add_scalar("Train/lr", optimizer.param_groups[-1]["lr"], epoch)
        optimizer.zero_grad()

        for batch_idx, (imgs, labels) in enumerate(train_dataloader):

            # 可视化网络、模型统计
            if epoch + batch_idx == 0:
                tb_writer.add_graph(vis_model, imgs.detach())
                summary(vis_model, imgs[0].unsqueeze(0).shape, device="cpu")
                del vis_model
            # 可视化增广图像
            if epoch % 10 + batch_idx == 0:
                category = [labels_list[label] for label in labels]
                vis_list = PreProcess().convert(imgs, category)
                for vis_name, vis_img in zip(set(category), vis_list):
                    tb_writer.add_image("Train/" + vis_name, vis_img, epoch)

            imgs, labels = imgs.to(device), labels.to(device)

            output = model(imgs)
            if TASK == "metric":
                hard_tuples = mining_func(output, labels)
                loss = loss_func(output, labels, hard_tuples)
            else:
                loss = loss_func(output, labels)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            ema_model.update(model)

            lr_scheduler.step_update(num_updates=epoch * len(train_dataloader) + batch_idx)

            if batch_idx % 100 == 0:
                iter_num = int(batch_idx + epoch * len(train_dataloader))
                tb_writer.add_scalar("Train/loss", loss.item(), iter_num)
        lr_scheduler.step(epoch + 1)

        # 验证集评估
        model.eval()
        if TASK == "class":  # 常规分类
            score = eval_model(model, val_dataloader).Overall_ACC
            ema_score = eval_model(ema_model.module, val_dataloader).Overall_ACC
            tb_writer.add_scalars("Eval", {"acc": score, "ema_acc": ema_score}, epoch)

        elif TASK == "metric":  # 度量学习
            score = eval_metric_model(model, train_set, val_set)
            ema_score = eval_metric_model(ema_model.module, train_set, val_set)
            tb_writer.add_scalars("Eval", {"precision": score, "ema_precision": ema_score}, epoch)
        model.train()

        # 保存最优模型
        score_dict = {score: model, ema_score: ema_model}
        max_score = max(score_dict)
        if best_score < max_score:
            best_score = max_score
            torch.save(score_dict[max_score], ckpt_path + "_best.pt")
    torch.save(model, ckpt_path + "_last.pt")
    torch.save(ema_model, ckpt_path + "_ema_last.pt")
    tb_writer.close()
