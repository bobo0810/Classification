import onnxruntime
import torch
import onnx
import onnxsim


class OnnxBackend:
    """
    onnx后端
    """

    def __init__(self):
        pass

    @staticmethod
    def convert(model, imgs, weights, dynamic, simplify):
        """
        torch转为onnx

        model: torch模型
        imgs: [B,C,H,W]Tensor
        weights: onnx权重保存路径
        dynamic: batch轴是否设为动态维度
        simplify: 是否简化onnx
        """
        torch.onnx.export(
            model,
            imgs,
            weights,
            verbose=False,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
            if dynamic
            else None,
        )
        model_onnx = onnx.load(weights)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        if simplify:
            try:
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={"input": list(imgs.shape)} if dynamic else None,
                )
                assert check, "assert check failed"
                onnx.save(model_onnx, weights)
            except Exception as e:
                print(f"simplifer failure: {e}")

        print("ONNX export success, saved as %s" % weights)
        print("\nVisualize onnx with https://github.com/lutzroeder/netron.")

    @staticmethod
    def infer(weights, imgs):
        """
        加载onnx并推理

        weights(str): onnx权重路径
        img(numpy): [B,C,H,W]
        """
        # 初始化
        session = onnxruntime.InferenceSession(
            weights, providers=["CPUExecutionProvider"]
        )
        # 推理
        output = session.run(
            [session.get_outputs()[0].name], {session.get_inputs()[0].name: imgs}
        )[0]
        return output
