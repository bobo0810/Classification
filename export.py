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
    parser.add_argument("--weights", help="模型权重", required=True) 

    # onnx
    parser.add_argument("--simplify", action="store_true", help="(可选)简化onnx")
    parser.add_argument("--dynamic", action="store_true", help="(可选)batch轴设为动态")

    # tensorrt
    parser.add_argument("--onnx2trt", action="store_true", help="(可选)onnx是否转为tensorrt")
    parser.add_argument("--fp16", action="store_true", help="(可选)开启fp16预测")
    cfg = parser.parse_args()

    # ==========================torch===============================
    imgs = torch.ones(tuple(cfg.img_size))
    # model = create_backbone(cfg.backbone, cfg.num_classes, checkpoint=cfg.weights) # 加载model.state_dict
    model = torch.load(cfg["Models"]["checkpoint"]) # 直接加载model
    model.eval()
    output_torch = model(imgs).detach().numpy()

    # ==========================导出ONNX===============================
    onnx_weights = cfg.weights.split(".")[0] + ".onnx"
    # torch转onnx
    OnnxBackend.convert(
        model=model,
        imgs=imgs,
        weights=onnx_weights,
        dynamic=cfg.dynamic,
        simplify=cfg.simplify,
    )
    # 加载模型并推理
    output_onnx = OnnxBackend.infer(weights=onnx_weights, imgs=imgs.numpy())

    # ==========================导出TensorRT===============================
    if cfg.onnx2trt: 
        assert cfg.dynamic == False, "Warn: only supported  fixed shapes"
        from Models.Backend.tensorrt import TensorrtBackend

        trt_weights = onnx_weights.split(".")[0] + ".trt"
        # onnx转tensorrt
        TensorrtBackend.convert(
            onnx_weights=onnx_weights,
            trt_weights=trt_weights,
            fp16=cfg.fp16,
        )
        # 加载模型并推理
        output_trt = TensorrtBackend.infer(
            weights=trt_weights, imgs=imgs.numpy(), output_shape=output_onnx.shape
        )

    # ==========================验证结果===============================
    print("\n", "*" * 28)
    print("output_torch - output_onnx = ", (output_torch - output_onnx).max())
    if cfg.onnx2trt:
        print("output_torch - output_trt = ", (output_torch - output_trt).max())
