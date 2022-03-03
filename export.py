import argparse
import os
from Models.Backbone import create_backbone
from Models.Backend.onnx import OnnxBackend
import torch

cur_path = os.path.abspath(os.path.dirname(__file__))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model")

    # torch
    parser.add_argument("--img_size", default=[1, 3, 224, 224], help="推理尺寸")
    parser.add_argument("--backbone", help="模型名称", default="resnet18")
    parser.add_argument("--weights", help="模型权重", required=True)
    parser.add_argument("--num_classes", type=int, help="类别数", required=True)

    # onnx
    parser.add_argument("--simplify", action="store_true", help="(可选)简化onnx")
    parser.add_argument("--dynamic", action="store_true", help="(可选)batch轴设为动态")
    cfg = parser.parse_args()

    # ==========================导出ONNX===============================
    imgs = torch.ones(tuple(cfg.img_size))
    # 加载torch模型
    model = create_backbone(cfg.backbone, cfg.num_classes, checkpoint=cfg.weights)
    model.eval()

    onnx_weights = cfg.weights.split(".")[0] + ".onnx"
    OnnxBackend.convert(
        model=model,
        imgs=imgs,
        weights=onnx_weights,
        dynamic=cfg.dynamic,
        simplify=cfg.simplify,
    )

    # ==========================验证===============================
    output_torch = model(imgs).detach().numpy()  # torch输出
    output_onnx = OnnxBackend.infer(weights=onnx_weights, imgs=imgs.numpy())  # onnx输出

    print("*" * 28)
    print("difference between torch and onnx is ", (output_torch - output_onnx).max())
