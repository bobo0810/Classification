"""
文件操作：分布式并行训练时，仅当rank=0的进程才可写入文件
"""
import os
import torch
import copy
import random
import time
import numpy as np
from collections import OrderedDict, Counter
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from Utils.tools import convert_vis
from bobotools.torch_tools import Torch_Tools
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
import colossalai

cur_path = os.path.abspath(os.path.dirname(__file__))


def copy_model(model):
    """
    复制模型，以便保存ddp模型
    """
    if gpc.get_global_rank() == 0:
        return copy.deepcopy(model)
    else:
        return None


def init_env(config):
    """
    初始化训练环境
    """
    # 固定随机种子
    seed = 227
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置CUDA
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # 初始化colossalai
    colossalai.launch_from_torch(config=config)

    # 日志
    exp_path = (
        os.path.dirname(cur_path)
        + "/ExpLog/"
        + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        + "/"
    )
    ckpt_path = os.path.join(exp_path, "checkpoint/")
    if not os.path.exists(ckpt_path) and gpc.get_global_rank() == 0:
        os.makedirs(ckpt_path)
    tb_writer = DDP_SummaryWriter(ckpt_path.replace("checkpoint/", "tb_log/"))
    tb_writer.add_text("Config", str(gpc.config))
    return ckpt_path, gpc.config, tb_writer, get_dist_logger()


def save_model(model, cp_model, ckpt_path):
    """
    保存模型权重
    因无法保存DDP模型，故DDP模型参数赋值给原生模型再保存。

    model: ddp封装的模型
    cp_model: 未封装的原生模型
    ckpt_path: 保存路径  eg:/home/xxx/xxx.pt
    """
    if gpc.get_global_rank() == 0:
        state_dict = model.state_dict()

        # 复制权重
        model_dict = cp_model.state_dict()
        new_state_dict = OrderedDict()
        # 遍历预训练参数
        for k, v in state_dict.items():
            name = k
            if "module." in name:
                name = name[7:]
            if "model." in name:
                name = name[6:]

            if name in model_dict:
                new_state_dict[name] = v
            else:
                # 不匹配
                print("error---->load pred_model mismatch:" + name)

        model_dict.update(new_state_dict)
        cp_model.load_state_dict(model_dict)
        # 保存
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        torch.save(cp_model, ckpt_path)


def save_criterion(criterion, ckpt_path):
    """
    保存分类器权重（度量学习）

    criterion: 分类器
    ckpt_path: 保存路径  eg:/home/xxx/xxx.pt
    """
    if gpc.get_global_rank() == 0:
        torch.save(criterion, ckpt_path)


class DDP_SummaryWriter:
    """
    分布式并行训练时，仅当rank=0的进程写入日志
    """

    def __init__(self, tb_path):
        """
        tb_path: 保存日志的路径
        """
        self.rank = gpc.get_global_rank()
        if self.rank == 0:
            self.tb_writer = SummaryWriter(tb_path)

    def add_text(self, tag, text_string):
        if self.rank == 0:
            self.tb_writer.add_text(tag, text_string)

    def add_scalar(self, tag, scalar_value, global_step):
        if self.rank == 0:
            self.tb_writer.add_scalar(tag, scalar_value, global_step)

    def add_model_info(self, model, size, batch=1, channel=3):
        """
        可视化模型信息
        size: 图像高、宽[224,224]
        """
        if self.rank == 0:
            input_shape = [batch, channel, size[0], size[1]]
            self.tb_writer.add_graph(model, torch.ones(size=input_shape))  # 可视化网络结构
            summary(model, input_shape, device="cpu")  # 打印网络信息

            time_dict = Torch_Tools.get_model_info(input_shape, model)  # 获取模型信息
            self.tb_writer.add_text("model info", str(time_dict))

    def add_dataset_info(self, dataset):
        """
        可视化数据集信息
        """
        if self.rank == 0:
            mode_list = ["train", "val", "test"]
            for mode in mode_list:
                info = "the total nums is %s" % (len(dataset[mode]["imgs"]))
                if "positive_pairs" in dataset[mode].keys():  # 样本对格式
                    info += "  \n positive_pairs: %s" % (
                        len(dataset[mode]["positive_pairs"])
                    )
                    info += "  \n negative_pairs: %s" % (
                        len(dataset[mode]["negative_pairs"])
                    )
                else:
                    info += "  \n labels is %s" % (dataset["all_labels"])
                    if "labels" in dataset[mode].keys():
                        info += "  \n  %s" % (dict(Counter(dataset[mode]["labels"])))
                self.tb_writer.add_text(mode, info)

    def add_augment_imgs(self, epoch, imgs, labels, labels_list):
        """
        可视化增广图像
        """
        if self.rank == 0 and epoch % 10 == 0:
            category = [labels_list[label] for label in labels]
            vis_list = convert_vis(imgs, category)
            for vis_name, vis_img in zip(set(category), vis_list):
                self.tb_writer.add_image("Train/" + vis_name, vis_img, epoch)

    def close(self):
        """
        关闭TensorBoard
        """
        if self.rank == 0:
            self.tb_writer.close()
