'''
Github Action 实现训练、推理
'''
import os
import argparse
import glob
cur_path = os.path.abspath(os.path.dirname(__file__))


def train():
    '''
    训练
    '''
    os.system('python train.py')
    print("Train Success!")

def predict():
    '''
    推理
    '''
    weights = glob.glob(cur_path+"/ExpLog/*/*/*.pt")[0]
    print("load weights from ",weights)
    os.system('python predict.py --weights %s  --vis_cam'%(weights))
    print("Predict Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Github Action")
    parser.add_argument("--mode",type=str, default="train",help="train|predict")
    args = parser.parse_args()

    assert args.mode in ["train","predict"]
    if args.mode == "train":
        train()
    elif args.mode == "predict":
        predict()
    else:
        print("Invalid mode")
