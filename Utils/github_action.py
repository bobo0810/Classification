"""
Github Action 实现训练、推理
"""
import os
import argparse
import glob

cur_path = os.path.abspath(os.path.dirname(__file__))


def train():
    """
    训练
    """
    os.system("python train.py")
    print("Train Success!")


def test():
    """
    测试
    """
    weights = glob.glob(cur_path + "/ExpLog/*/*/*.pt")[0]
    print("load weights from ", weights)

    os.system("python test.py --weights %s " % (weights))
    print("Test Success!")


def predict():
    """
    推理
    """
    weights = glob.glob(cur_path + "/ExpLog/*/*/*.pt")[0]
    print("load weights from ", weights)

    os.system("python predict.py --weights %s  --vis_cam" % (weights))
    print("Predict Success!")


def deploy():
    """
    部署 torch->torchscript/onnx
    """
    weights = glob.glob(cur_path + "/ExpLog/*/*/*.pt")[0]
    print("load weights from ", weights)

    os.system(
        "python export.py --weights %s  --torch2script  --torch2onnx " % (weights)
    )
    print("Deploy Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Github Action")
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    assert args.mode in ["train", "test", "predict", "deploy"]
    method = {"train": train, "test": test, "predict": predict, "deploy": deploy}
    method[args.mode]()

