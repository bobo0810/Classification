import argparse
import os
from Models.Backbone import create_backbone
import torch
import onnx
import onnxsim

cur_path = os.path.abspath(os.path.dirname(__file__))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument("--img_size", default=[1, 3, 224, 224], help="推理尺寸")
    # 模型
    parser.add_argument("--backbone", help="模型名称", default="resnet18")
    parser.add_argument("--weights", help="模型权重", required=True)
    parser.add_argument("--num_classes", type=int, help="类别数", required=True)
    # onnx
    parser.add_argument("--simplify", action="store_true", help="(可选)简化onnx,默认关闭")
    parser.add_argument("--dynamic", action="store_true", help="(可选)batch轴设为动态,默认关闭")
    cfg = parser.parse_args()

    # 加载模型
    model = create_backbone(cfg.backbone, cfg.num_classes, checkpoint=cfg.weights)
    model.eval()

    # 输入图像
    img = torch.ones(tuple(cfg.img_size))
    print("output shape is ", model(img).shape)

    # ==========================导出ONNX===============================
    print("\n onnx version is %s" % onnx.__version__)
    onnx_weights = cfg.weights.split(".")[0] + ".onnx"
    torch.onnx.export(
        model,
        img,
        onnx_weights,
        verbose=False,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
        if cfg.dynamic
        else None,
    )

    model_onnx = onnx.load(onnx_weights)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    if cfg.simplify:
        try:
            print(f" onnx-simplifier {onnxsim.__version__}...")
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=cfg.dynamic,
                input_shapes={"input": list(img.shape)} if cfg.dynamic else None,
            )
            assert check, "assert check failed"
            onnx.save(model_onnx, onnx_weights)
        except Exception as e:
            print(f"simplifer failure: {e}")

    print("ONNX export success, saved as %s" % onnx_weights)
    print("\nVisualize onnx with https://github.com/lutzroeder/netron.")
