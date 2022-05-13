from torch.utils.tensorboard import SummaryWriter
import os
class SummaryWriter_DDP:
    '''
    分布式并行训练时，仅当rank=0的进程写入日志
    '''
    def __init__(self,tb_path,rank):
        '''
        tb_path: 保存日志的路径
        rank: 当前进程号
        '''
        self.rank=rank
        if self.rank==0:
            self.tb_writer = SummaryWriter(tb_path)
    
    def add_text(self,tag,text_string):
        if self.rank==0:
            self.tb_writer.add_text(tag,text_string)

    
    def add_scalar(self,tag,scalar_value,global_step):
        if self.rank==0:
            self.tb_writer.add_scalar(tag,scalar_value,global_step)

    
    def close(self):
        if self.rank==0:
            self.tb_writer.close()

