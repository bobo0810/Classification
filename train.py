import sys
import os
import torch
import argparse
import yaml
from DataSets import create_dataloader
from DataSets.preprocess import PreProcess
from Utils.tools import init_env, get_category, eval_model, eval_metric_model
from Models.Backbone import create_backbone
from Models.Loss import create_class_loss, create_metric_loss
from Models.Optimizer import create_optimizer
from Models.Scheduler import create_scheduler
from Utils.tools import tensor2img
from timm.utils import ModelEmaV2
from torchinfo import summary
from DataSets.dataset import create_datasets
from pytorch_metric_learning import losses

cur_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--yaml", help="训练配置", default=cur_path + "/Config/train.yaml")
    parser.add_argument("--txt", help="训练集路径", default=cur_path + "/Config/train.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    file = open(args.yaml, "r")
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    cfg["DataSet"]["txt"] = args.txt
    cfg["DataSet"]["labels"] = get_category(
        path=os.path.dirname(args.txt) + "/labels.txt"
    )

    # 初始化环境
    tb_writer, checkpoint_path = init_env(cfg)

    # 模型
    model = create_backbone(
        cfg["Models"]["backbone"],
        num_classes=len(cfg["DataSet"]["labels"]),
    )
    TASK = "metric" if hasattr(model, "embedding_size") else "class"
    # 区分任务
    if TASK == "metric":

        # 损失函数(分类器)
        loss_func = create_metric_loss(
            name=cfg["Models"]["loss"],
            num_classes=len(cfg["DataSet"]["labels"]),
            embedding_size=model.embedding_size,
        ).to(device)
        params = [{"params": loss_func.parameters()}]

        # 用于验证度量指标
        train_set = create_datasets(cfg["DataSet"], mode="train")
        val_set = create_datasets(cfg["DataSet"], mode="val")

    else:
        # 损失函数
        loss_func = create_class_loss(cfg["Models"]["loss"]).to(device)
        params = []

    # 模型转为GPU
    if device != "cpu":
        model = torch.nn.DataParallel(model).to(device)
    model.train()
    ema_model = ModelEmaV2(model, decay=0.9998)

    # 优化器
    params.append({"params": model.parameters()})
    optimizer = create_optimizer(
        params, cfg["Models"]["optimizer"], lr=cfg["Train"]["lr"]
    )

    # 数据集
    train_dataloader = create_dataloader(cfg["DataSet"], mode="train")
    val_dataloader = create_dataloader(cfg["DataSet"], mode="val")

    # 学习率调度器
    lr_scheduler = create_scheduler(
        sched_name=cfg["Train"]["scheduler"],
        epochs=cfg["Train"]["epochs"],
        optimizer=optimizer,
    )
    best_acc = 0.0
    for epoch in range(cfg["Train"]["epochs"]):
        print("start epoch {}/{}...".format(epoch, cfg["Train"]["epochs"]))
        tb_writer.add_scalar("Train/lr", optimizer.param_groups[-1]["lr"], epoch)
        optimizer.zero_grad()

        for batch_idx, (imgs, labels) in enumerate(train_dataloader):

            # 可视化网络、模型统计
            if epoch + batch_idx == 0:
                tb_writer.add_graph(model, imgs)
                summary(model, imgs[0].unsqueeze(0).shape, device=device)
            # 可视化增广图像
            if epoch % 10 + batch_idx == 0:
                category = [cfg["DataSet"]["labels"][label] for label in labels]
                vis_list = PreProcess().convert(imgs, category)
                for vis_name, vis_img in zip(set(category), vis_list):
                    tb_writer.add_image("Train/" + vis_name, vis_img, epoch)

            imgs, labels = imgs.to(device), labels.to(device)

            output = model(imgs)
            loss = loss_func(output, labels)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            ema_model.update(model)

            lr_scheduler.step_update(
                num_updates=epoch * len(train_dataloader) + batch_idx
            )

            if batch_idx % 100 == 0:
                iter_num = int(batch_idx + epoch * len(train_dataloader))
                tb_writer.add_scalar("Train/loss", loss.item(), iter_num)
        lr_scheduler.step(epoch + 1)

        # 验证集评估
        model.eval()
        if TASK == "class":  # 常规分类
            acc, _ = eval_model(model, val_dataloader)
            ema_acc, _ = eval_model(ema_model.module, val_dataloader)
            tb_writer.add_scalars("Eval", {"acc": acc, "ema_acc": ema_acc}, epoch)

        elif TASK == "metric":  # 度量学习
            acc = eval_metric_model(model, train_set, val_set)
            ema_acc = eval_metric_model(ema_model.module, train_set, val_set)
            tb_writer.add_scalars(
                "Eval", {"precision": acc, "ema_precision": ema_acc}, epoch
            )
        model.train()

        # 保存最优模型
        acc_dict = {acc: model, ema_acc: ema_model}
        max_acc = max(acc_dict)
        if best_acc < max_acc:
            best_acc = max_acc
            torch.save(acc_dict[max_acc], checkpoint_path + "_best.pt")
    torch.save(model, checkpoint_path + "_last.pt")
    torch.save(ema_model, checkpoint_path + "_ema_last.pt")
    tb_writer.close()
