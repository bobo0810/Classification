import sys
import os
import torch
from DataSets import create_dataloader
from DataSets.preprocess import PreProcess
from Utils.tools import init_env, eval_metric, get_labels
from Models.Backbone import create_backbone
from Models.Loss import create_loss
from Models.Optimizer import create_optimizer
from Models.Scheduler import create_scheduler
from timm.utils import ModelEmaV2
from torchinfo import summary
import argparse
import yaml

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
    cfg["DataSet"]["labels"] = get_labels(
        path=os.path.dirname(args.txt) + "/labels.txt"
    )

    # 初始化环境
    tb_writer, checkpoint_path = init_env(cfg)

    # 模型
    model = create_backbone(
        cfg["Models"]["backbone"],
        num_classes=len(cfg["DataSet"]["labels"]),
    )
    task = model.task
    if not device == "cpu":
        model = torch.nn.DataParallel(model).to(device)
    model.train()
    ema_model = ModelEmaV2(model, decay=0.9998)

    # 损失函数
    criterion = create_loss(cfg["Models"]["loss"])

    # 优化器
    optimizer = create_optimizer(
        model, cfg["Models"]["optimizer"], lr=cfg["Train"]["lr"]
    )

    # 学习率调度器
    lr_scheduler = create_scheduler(
        sched_name=cfg["Train"]["scheduler"],
        epochs=cfg["Train"]["epochs"],
        optimizer=optimizer,
    )

    # 数据集
    train_dataloader = create_dataloader(cfg["DataSet"], mode="train")
    val_dataloader = create_dataloader(cfg["DataSet"], mode="val")

    best_acc, best_ema_acc = 0.0, 0.0
    for epoch in range(cfg["Train"]["epochs"]):
        print("start epoch {}/{}...".format(epoch, cfg["Train"]["epochs"]))
        tb_writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)
        optimizer.zero_grad()

        for batch_idx, (imgs, labels, names) in enumerate(train_dataloader):

            if epoch + batch_idx == 0 and task == "class":
                tb_writer.add_graph(model, imgs.clone())  # 网络结构可视化
                summary(model, imgs[0].unsqueeze(0).shape, device=device)  # 模型统计
            # 图像可视化
            if epoch % 10 + batch_idx == 0:
                vis_list = PreProcess().convert(imgs, names)
                for vis_name, vis_img in zip(set(names), vis_list):
                    tb_writer.add_image("Train/" + vis_name, vis_img, epoch)

            imgs, labels = imgs.to(device), labels.to(device)
            if task == "class":  # 常规分类
                output = model(imgs)
                loss = criterion(output, labels)
            elif task == "metric":  # 度量学习
                loss = model(imgs, labels)
            else:
                raise NotImplementedError

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
        if task == "class":
            model.eval()
            acc, _ = eval_metric(model, val_dataloader, device)
            ema_acc, _ = eval_metric(ema_model.module, val_dataloader, device)
            tb_writer.add_scalars("Eval", {"acc": acc, "ema_acc": ema_acc}, epoch)
            model.train()

            if best_acc < acc:
                best_acc = acc
                torch.save(
                    model,  # model.state_dict()
                    checkpoint_path + cfg["Models"]["backbone"] + "_best.pt",
                )
            if best_ema_acc < ema_acc:
                best_ema_acc = ema_acc
                torch.save(
                    ema_model,
                    checkpoint_path + cfg["Models"]["backbone"] + "_best_ema.pt",
                )

    torch.save(
        model,
        checkpoint_path + cfg["Models"]["backbone"] + "_last.pt",
    )
    torch.save(
        ema_model,
        checkpoint_path + cfg["Models"]["backbone"] + "_last_ema.pt",
    )
    tb_writer.close()
