import openvino.inference_engine as ie
import subprocess
from pathlib import Path


class OpennVINOBackend:
    """
    OpennVINO后端
    """

    def __init__(self):
        pass

    @staticmethod
    def convert(onnx_weights, weights):
        """
        onnx模型转为OpennVINO模型

        onnx_weights: onnx模型路径
        weights: OpennVINO文件的保存路径
        """

        try:
            cmd = f"mo --input_model {onnx_weights} --output_dir {weights}"
            subprocess.check_output(cmd, shell=True)
            print("*" * 28)
            print("OpennVINO export success, saved as %s" % weights)
        except Exception as e:
            print("OpennVINO export failure: {e}")

    @staticmethod
    def infer(weights, imgs):
        """
        加载OpennVINO并推理

        weights(str): onnx权重路径
        img(numpy): [B,C,H,W]
        """
        # 加载模型
        core = ie.IECore()
        if not Path(weights).is_file():
            w = next(Path(weights).glob("*.xml"))
        network = core.read_network(
            model=w, weights=Path(w).with_suffix(".bin")
        )  # *.xml, *.bin paths
        executable_network = core.load_network(
            network, device_name="CPU", num_requests=1
        )

        # 推理
        desc = ie.TensorDesc(
            precision="FP32", dims=imgs.shape, layout="NCHW"
        )  # Tensor Description
        request = executable_network.requests[0]  # inference request
        request.set_blob(blob_name="input", blob=ie.Blob(desc, imgs))
        request.infer()
        name = next(iter(request.output_blobs))
        output = request.output_blobs["output"].buffer

        # name=next(iter(request.input_blobs))
        # name=next(iter(request.output_blobs))
        return output
