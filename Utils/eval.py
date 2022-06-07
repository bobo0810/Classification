import torch
from pycm import ConfusionMatrix
from DataSets import create_datasets, create_dataloader
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import numpy as np


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
def eval_metric_model(model, dataset, img_size, process_name, batch_size, mode):
    """
    度量学习：评估指标

    model: 模型
    dataset: 数据集信息
    img_size: 图像尺寸
    process_name: 图像预处理的名称
    batch_size: 推理批次
    mode: 指定评估类型
    """
    assert mode in ["val", "test"]
    if "positive_pairs" in dataset[mode].keys():
        # [类型,是否为同类,图片1,图片2] 样本对格式,统计误识率FPR下的通过率TPR
        val_set = create_datasets(
            dataset=dataset[mode], size=img_size, process=process_name
        )
        val_dataloader = create_dataloader(batch_size, val_set)
        # 获得特征
        device = torch.device("cuda:0")
        img_to_feature = get_feature(val_dataloader, model, device, use_mirror=False)
        # 计算余弦分数
        positive_score, negative_score = get_score(
            img_to_feature,
            dataset[mode]["positive_pairs"],
            dataset[mode]["negative_pairs"],
        )
        # 统计
        FPR_List, TPR_List = cal_index(positive_score, negative_score)
        print("FPR_List", FPR_List)
        print("TPR_List", TPR_List)
        return TPR_List[1]  # 1e-4误识下的通过率

    else:
        # [类型,类别名,图像路径]格式，统计精确率
        # 原理：训练集的某类所有图片特征的均值作为该类的特征中心，测试样本与某类距离最近即判定为该类。
        train_set = create_datasets(
            dataset=dataset["train"], size=img_size, process=process_name
        )
        val_set = create_datasets(
            dataset=dataset[mode], size=img_size, process=process_name
        )

        tester = testers.BaseTester(batch_size=batch_size, dataloader_num_workers=4)
        train_embeddings, train_labels = tester.get_all_embeddings(train_set, model)
        test_embeddings, test_labels = tester.get_all_embeddings(val_set, model)
        train_labels, test_labels = train_labels.squeeze(1), test_labels.squeeze(1)

        accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
        accuracies = accuracy_calculator.get_accuracy(
            test_embeddings, train_embeddings, test_labels, train_labels, False
        )
        precision = accuracies["precision_at_1"]
        return precision


@torch.no_grad()
def get_feature(
    dataloader,
    model,
    device,
    use_mirror=False,
):
    """
    提取特征
    """
    img_to_feature = {}
    for i, (img, img_path) in enumerate(dataloader):
        img = img.to(device)
        feature_npy = model(img).detach().cpu().numpy()
        if use_mirror:
            feature_npy += model(img.flip(-1)).detach().cpu().numpy()
        # 保存图像及对应特征
        for j in range(len(img_path)):
            img_to_feature[img_path[j]] = feature_npy[j]
    return img_to_feature


def get_score(img_to_feature, positive_pairs, negative_pairs):
    """
    根据特征结果 和 正负样本对，计算余弦分数
    """

    # 保存 正样本对 每条记录的余弦相似度
    positive_score = []
    for img1, img2 in positive_pairs:
        feature_1 = img_to_feature[img1]
        feature_2 = img_to_feature[img2]
        # 计算 证件照和生活照的特征结果 的余弦相似度
        positive_score.append(
            np.inner(feature_1, feature_2)
            / np.power(np.sum(np.power(feature_1, 2)), 0.5)
            / np.power(np.sum(np.power(feature_2, 2)), 0.5)
        )

    # 保存 负样本对 每条记录的余弦相似度
    negative_score = []
    for img1, img2 in negative_pairs:
        feature_1 = img_to_feature[img1]
        feature_2 = img_to_feature[img2]
        negative_score.append(
            np.inner(feature_1, feature_2)
            / np.power(np.sum(np.power(feature_1, 2)), 0.5)
            / np.power(np.sum(np.power(feature_2, 2)), 0.5)
        )
    return positive_score, negative_score


def cal_index(positive_score, negative_score):

    P = len(positive_score)
    N = len(negative_score)
    score_cosin = positive_score
    score_cosin.extend(negative_score)
    label = [1] * P
    label.extend([0] * N)
    score = sorted(score_cosin, reverse=True)
    index = np.argsort(-np.array(score_cosin))
    label_sort = []
    for i in range(len(index)):
        label_sort.append(label[index[i]])

    TPR = []
    FPR = []

    # 二万分之一、万分之一、千分之一 误识率及对应通过率
    FPR_List = [0.0, 0.0, 0.0]
    TPR_List = [0.0, 0.0, 0.0]

    for idx in range(len(score)):
        FN = P - np.array(label_sort[0 : idx + 1]).sum()
        FP = idx + 1 - np.array(label_sort[0 : idx + 1]).sum()
        false_accept_rate = FP / N
        false_reject_rate = FN / P
        TPR.append(1 - false_reject_rate)
        FPR.append(false_accept_rate)
        if FPR[idx] > 0.00005 and FPR_List[0] == 0.0:
            FPR_List[0] = FPR[idx]
            TPR_List[0] = TPR[idx]
        if FPR[idx] > 0.0001 and FPR_List[1] == 0.0:
            FPR_List[1] = FPR[idx]
            TPR_List[1] = TPR[idx]
        if FPR[idx] > 0.001 and FPR_List[2] == 0.0:
            FPR_List[2] = FPR[idx]
            TPR_List[2] = TPR[idx]
            break
    # 二万分之一、万分之一、千分之一 误识率下的通过率
    return FPR_List, TPR_List
