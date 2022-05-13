from torch.utils.tensorboard import SummaryWriter
import os
import torch
from torchinfo import summary
from Utils.tools import convert_vis

class SummaryWriter_DDP:
    """
    分布式并行训练时，仅当rank=0的进程写入日志
    """

    def __init__(self, tb_path, rank):
        """
        tb_path: 保存日志的路径
        rank: 当前进程号
        """
        self.rank = rank
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
