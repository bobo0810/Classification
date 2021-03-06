import torch
from pycm import ConfusionMatrix
from DataSets import create_datasets, create_dataloader
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import numpy as np
from .tools import cal_index, get_score, get_all_embeddings


@torch.no_grad()
def eval_model(model, data_loader):
    """
    常规分类：评估指标
    """
    preds_list, labels_list = [], []
    for batch_idx, (imgs, labels) in enumerate(data_loader):
        imgs, labels = imgs.cuda(), labels.cuda()
        scores = model(imgs)
        scores = torch.nn.functional.softmax(scores, dim=1)
        preds = torch.argmax(scores, dim=1)

        preds_list.append(preds)
        labels_list.append(labels)
    preds_list = torch.cat(preds_list, dim=0).cpu().numpy()
    labels_list = torch.cat(labels_list, dim=0).cpu().numpy()

    # 统计
    return ConfusionMatrix(labels_list, preds_list)


@torch.no_grad()
def eval_metric_model(
    model, dataset, img_size, process_name, batch_size, mode, mirror=False
):
    """
    度量学习：评估指标

    model: 模型
    dataset: 数据集信息
    img_size: 图像尺寸
    process_name: 图像预处理的名称
    batch_size: 推理批次
    mode: 指定评估类型
    mirror: 融合镜像特征，默认关闭
    """
    assert mode in ["val", "test"]

    if "positive_pairs" in dataset[mode].keys():
        # [类型,是否为同类,图片1,图片2] 样本对格式,统计误识率FPR下的通过率TPR
        val_set = create_datasets(
            dataset=dataset[mode], size=img_size, process=process_name
        )
        val_dataloader = create_dataloader(batch_size, val_set)
        # 获得特征
        img_feature_dict = get_all_embeddings(val_dataloader, model, mirror)
        # 计算余弦分数
        positive_score, negative_score = get_score(
            img_feature_dict,
            dataset[mode]["positive_pairs"],
            dataset[mode]["negative_pairs"],
        )
        # 统计误识率下的通过率
        FPR_TPR = cal_index(positive_score, negative_score)
        if mode == "val":
            return {"index": "1e-4FPR", "value": FPR_TPR[0.0001]}
        else:
            return {"1e-4FPR": FPR_TPR[0.0001], "1e-3FPR": FPR_TPR[0.001]}

    else:
        # [类型,类别名,图像路径]格式，统计精确率
        # 原理：训练集的某类所有图片特征的均值作为该类的特征中心，测试样本与某类距离最近即判定为该类。
        train_set = create_datasets(
            dataset=dataset["train"], size=img_size, process=process_name
        )
        val_set = create_datasets(
            dataset=dataset[mode], size=img_size, process=process_name
        )

        tester = testers.BaseTester(batch_size=batch_size, dataloader_num_workers=8)
        train_embeddings, train_labels = tester.get_all_embeddings(train_set, model)
        test_embeddings, test_labels = tester.get_all_embeddings(val_set, model)
        train_labels, test_labels = train_labels.squeeze(1), test_labels.squeeze(1)

        accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
        accuracies = accuracy_calculator.get_accuracy(
            test_embeddings, train_embeddings, test_labels, train_labels, False
        )
        precision = accuracies["precision_at_1"]
        if mode == "val":
            return {"index": "precision", "value": precision}
        else:
            return {"precision": precision}
