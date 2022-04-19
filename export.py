import argparse
import os
import torch

cur_path = os.path.abspath(os.path.dirname(__file__))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model")

    # torch
    parser.add_argument("--img_size", default=[1, 3, 224, 224], help="推理尺寸")
    parser.add_argument("--weights", help="模型权重", required=True)

    # torchscript
    parser.add_argument("--torch2script", action="store_true", help="(可选)转为torchscript")
    parser.add_argument("--script_gpu", action="store_true", help="(可选)导出GPU模型，默认CPU模型")

    # onnx
    parser.add_argument("--torch2onnx", action="store_true", help="(可选)转为onnx")
    parser.add_argument("--simplify", action="store_true", help="(可选)简化onnx")
    parser.add_argument("--dynamic", action="store_true", help="(可选)batch轴设为动态")

    # tensorrt
    parser.add_argument("--onnx2trt", action="store_true", help="(可选)转为tensorrt")
    parser.add_argument("--fp16", action="store_true", help="(可选)开启fp16预测")

    # openvino
    parser.add_argument("--onnx2openvino", action="store_true", help="(可选)转为openvino")

    cfg = parser.parse_args()

    # ==========================torch===============================
    imgs = torch.ones(tuple(cfg.img_size))
    model = torch.load(cfg.weights, map_location="cpu")  # 直接加载model，而非model.state_dict
    while hasattr(model, "module"):
        model = model.module
    model.eval()
    output_torch = model(imgs).detach().numpy()
    # ==========================导出TorchScript===============================
    if cfg.torch2script:
        from Models.Backend.torchscript import ScriptBackend

        if cfg.script_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        script_weights = cfg.weights.split(".")[0] + ".torchscript"
        ScriptBackend.convert(
            model=model.to(device),
            imgs=imgs.to(device),
            weights=script_weights,
        )
        output_script = ScriptBackend.infer(
            weights=script_weights, imgs=imgs.to(device)
        )
        output_script = output_script.detach().cpu().numpy()

    # ==========================导出ONNX===============================
    if cfg.torch2onnx:
        from Models.Backend.onnx import OnnxBackend

        onnx_weights = cfg.weights.split(".")[0] + ".onnx"
        # torch转onnx
        OnnxBackend.convert(
            model=model,
            imgs=imgs,
            weights=onnx_weights,
            dynamic=cfg.dynamic,
            simplify=cfg.simplify,
        )
        output_onnx = OnnxBackend.infer(weights=onnx_weights, imgs=imgs.numpy())

    # ==========================导出TensorRT===============================
    if cfg.onnx2trt:
        assert cfg.dynamic == False, "Warn: only supported  fixed shapes"
        assert os.path.exists(onnx_weights), "Warn: %s no exist" % onnx_weights
        from Models.Backend.tensorrt import TensorrtBackend

        trt_weights = onnx_weights.split(".")[0] + ".trt"
        # onnx转tensorrt
        TensorrtBackend.convert(
            onnx_weights=onnx_weights,
            trt_weights=trt_weights,
            fp16=cfg.fp16,
        )
        output_trt = TensorrtBackend.infer(
            weights=trt_weights, imgs=imgs.numpy(), output_shape=output_onnx.shape
        )

    # ==========================导出OpenVINO===============================
    if cfg.onnx2openvino:
        assert cfg.dynamic == False, "Warn: only supported  fixed shapes"
        assert os.path.exists(onnx_weights), "Warn: %s no exist" % onnx_weights
        openvino_weights = onnx_weights.split(".")[0] + "_openvino"
        from Models.Backend.openvino import OpennVINOBackend

        # onnx转openvino
        OpennVINOBackend.convert(onnx_weights, openvino_weights)
        output_openvino = OpennVINOBackend.infer(
            weights=openvino_weights, imgs=imgs.numpy()
        )
    # ==========================验证结果===============================
    print("\n", "*" * 28)
    if cfg.torch2script:
        print("output_torch - output_script = ", (output_torch - output_script).max())
    if cfg.torch2onnx:
        print("output_torch - output_onnx = ", (output_torch - output_onnx).max())
    if cfg.onnx2trt:
        print("output_torch - output_trt = ", (output_torch - output_trt).max())
    if cfg.onnx2openvino:
        print(
            "output_torch - output_openvino = ", (output_torch - output_openvino).max()
        )
