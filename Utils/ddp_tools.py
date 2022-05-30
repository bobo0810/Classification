"""
文件操作：分布式并行训练时，仅当rank=0的进程才可写入文件
"""
import os
import torch
import copy
import random
import time
import numpy as np
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from Utils.tools import convert_vis
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


def init_env(config_file):
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
    colossalai.launch_from_torch(config=config_file)

    # 日志路径
    exp_path = (
        os.path.dirname(cur_path)
        + "/ExpLog/"
        + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        + "/"
    )
    ckpt_path = os.path.join(exp_path, "checkpoint/")
    tb_path = os.path.join(exp_path, "tb_log/")
    if not os.path.exists(ckpt_path) and gpc.get_global_rank() == 0:
        os.makedirs(ckpt_path)
    return ckpt_path, tb_path, gpc.config, get_dist_logger()


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

    def add_graph(self, model, size, batch=1, channel=3):
        """
        可视化模型结构
        size: 图像高、宽[224,224]
        """
        if self.rank == 0:
            input_shape = [batch, channel, size[0], size[1]]
            self.tb_writer.add_graph(model, torch.ones(size=input_shape))

            # 模型统计
            summary(model, input_shape, device="cpu")

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
        if self.rank == 0:
            self.tb_writer.close()
