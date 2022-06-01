import torch
from pycm import ConfusionMatrix
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

@torch.no_grad()
def eval_metric_model(model, train_set, val_set, batch_size):
    """
    度量学习：评估指标
    """
    tester = testers.BaseTester(batch_size=batch_size, dataloader_num_workers=4)
    train_embeddings, train_labels = tester.get_all_embeddings(train_set, model)
    test_embeddings, test_labels = tester.get_all_embeddings(val_set, model)
    train_labels, test_labels = train_labels.squeeze(1), test_labels.squeeze(1)

    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    return accuracies["precision_at_1"]


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
