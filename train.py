import sys
import os
import torch
from DataSets import DataSets
from Utils.tools import init_env, eval_confusion_matrix
import yaml
from Models.Backbone import Backbone
from Models.Head import Head
from Models.Optimizer import Optimizer

cur_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    device = torch.device("cuda:0")  # 'cuda:0'

    file = open(cur_path + "/Config/train.yaml", "r")
    cfg = yaml.load(file, Loader=yaml.FullLoader)

    # 初始化环境
    tb_writer, checkpoint_path = init_env(cfg)

    # 模型
    model = Backbone(
        cfg["Models"]["backbone"],
        class_nums=len(cfg["DataSet"]["category"]),
    )
    model.to(device)
    model.eval()

    # 损失函数
    criterion = Head(cfg["Models"]["head"])

    # 优化器
    optimizer = Optimizer(model, cfg["Models"]["optimizer"], lr=cfg["Train"]["lr"])

    # 学习率
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["Train"]["milestones"], gamma=0.1
    )

    # 数据集
    train_dataloader = DataSets(cfg["DataSet"], mode="train")
    val_dataloader = DataSets(cfg["DataSet"], mode="val")

    for epoch in range(cfg["Train"]["epochs"]):
        print("start epoch {}/{}...".format(epoch, cfg["Train"]["epochs"]))

        tb_writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

        optimizer.zero_grad()
        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()

            if cfg["Models"]["optimizer"] == "SGD":
                optimizer.step()
                optimizer.zero_grad()
            elif cfg["Models"]["optimizer"] == "SAM":
                optimizer.first_step(zero_grad=True)
                criterion(model(imgs), labels).backward()
                optimizer.second_step(zero_grad=True)
            else:
                raise NotImplementedError

            if batch_idx % 100 == 0:
                iter_num = int(batch_idx + epoch * len(train_dataloader))
                tb_writer.add_scalar("Train/loss", loss.item(), iter_num)
        # 保存
        torch.save(
            model.state_dict(),
            checkpoint_path + cfg["Models"]["backbone"] + "_" + "%03d" % epoch + ".pth",
        )
        # 评估
        model.eval()
        acc = eval_confusion_matrix(model, val_dataloader, device).Overall_ACC
        tb_writer.add_scalar("Eval/acc", acc, epoch)
        model.train()
        lr_scheduler.step()
    tb_writer.close()
