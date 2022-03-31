import argparse
import os
import torch
import cv2
from PIL import Image
from timm.data.transforms_factory import create_transform as timm_transform
from Utils.tools import vis_cam

cur_path = os.path.abspath(os.path.dirname(__file__))


def img_preprocess(img_path, img_size):
    """
    图像预处理

    img_path(str): 图像路径。eg:/home/aaa.jpg
    img_size(list): 图像尺寸。eg:[224, 224]

    """
    assert os.path.exists(img_path), "图像文件不存在"
    bgr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.fromarray(bgr_img)
    img_tensor = timm_transform(img_size)(img)
    return img_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict image")
    # 图像
    parser.add_argument(
        "--img_path",
        type=str,
        default=cur_path + "/CatDog/cat/cat_1.jpg",
        help="Input image path",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default=cur_path + "/Config/labels.txt",
        help="labels.txt path",
    )

    # 模型
    parser.add_argument("--img_size", default=[224, 224], help="推理尺寸")
    parser.add_argument("--weights", type=str, help="模型权重", required=True)
    # 可视化注意力图
    parser.add_argument("--vis_cam", action="store_true", help="可视化注意力图")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 类别
    labels = open(args.labels_path, "r").readlines()
    labels = [line.strip() for line in labels if not line.strip() == ""]

    # 图像预处理
    img_tensor = img_preprocess(img_path=args.img_path, img_size=args.img_size)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 加载模型
    model = torch.load(args.weights, map_location="cpu")
    model = model.module if model.module else model
    model.to(device)
    model.eval()

    # 推理
    scores = model(img_tensor)
    scores = torch.nn.functional.softmax(scores, dim=1)

    score_sort, idx_sort = torch.sort(scores, dim=1, descending=True)
    score_sort, idx_sort = score_sort[0], idx_sort[0]  # batch=1

    pred_labels = labels[idx_sort[0]]
    pred_probs = score_sort[0]
    print(" %s, %s , %f" % (args.img_path, pred_labels, pred_probs.item()))

    # 可视化注意力图
    if args.vis_cam:
        cam_image = vis_cam(model, img_tensor)
        save_path = cur_path + "/cam_img.jpg"
        cv2.imwrite(save_path, cam_image)
        print("cam_image are generated in ", save_path)
