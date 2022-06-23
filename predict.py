import argparse
import os
import torch
import cv2
from PIL import Image
from Utils.tools import vis_cam
from DataSets.preprocess import preprocess

cur_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict image")
    # 图像
    parser.add_argument(
        "--img_path",
        type=str,
        default=cur_path + "/CatDog/cat/cat_1.jpg",
        help="测试图片路径",
    )
    parser.add_argument("--process", help="图像预处理", default="ImageNet")
    # 模型
    parser.add_argument("--size", type=str, help="图像宽高", default="224,224")
    parser.add_argument("--weights", type=str, help="模型权重", required=True)
    # 可视化注意力图
    parser.add_argument("--vis_cam", action="store_true", help="可视化注意力图,默认关闭(仅分类任务生效)")

    args = parser.parse_args()
    args.size = [int(line) for line in args.size.split(",")]

    # 图像预处理
    img = preprocess(
        args.process, args.img_path, args.size, use_augment=False
    ).unsqueeze(0)

    # 加载模型
    model = torch.load(args.weights, map_location="cpu")
    model.eval()

    if model.info["task"] == "metric":
        # 度量学习，仅输出特征
        feature = model(img)
        print("feature: ", feature)
    else:
        # 分类任务: 输出类别概率、注意力图
        labels = model.info["all_labels"]
        # 推理
        scores = model(img)
        scores = torch.nn.functional.softmax(scores, dim=1)

        score_sort, idx_sort = torch.sort(scores, dim=1, descending=True)
        score_sort, idx_sort = score_sort[0], idx_sort[0]  # batch=1

        pred_labels = labels[idx_sort[0]]
        pred_probs = score_sort[0]
        print(" %s, %s , %f" % (args.img_path, pred_labels, pred_probs.item()))

        # 可视化注意力图
        if args.vis_cam:
            cam_image = vis_cam(model, img)
            save_path = cur_path + "/cam_img.jpg"
            cv2.imwrite(save_path, cam_image)
            print("cam_image are generated in ", save_path)
