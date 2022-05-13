"""
文件操作：分布式并行训练时，仅当rank=0的进程才可写入文件
"""
import os
import torch
import copy
from collections import OrderedDict

def create_folder(path, rank):
    """
    创建指定文件夹
    """
    if not os.path.exists(path) and rank == 0:
        os.makedirs(path)

def copy_model(model,rank):
    if rank==0:
        return copy.deepcopy(model)
    else:
        return None

def save_model(model, cp_model,ckpt_path, rank):
    """
    保存模型权重 
    因无法保存DDP模型，故DDP模型参数赋值给原生模型再保存。
    
    model: ddp封装的模型
    cp_model: 未封装的原生模型
    ckpt_path: 保存路径  eg:/home/xxx/xxx.pt
    """
    if rank==0:
        state_dict=model.state_dict()

        # 复制权重
        model_dict = cp_model.state_dict() 
        new_state_dict = OrderedDict()
        # 遍历预训练参数
        for k, v in state_dict.items():
            name = k
            if 'module.' in name:
                name = name[7:]
            if 'model.' in name:
                name = name[6:]
            
            if name in model_dict:
                new_state_dict[name] = v
            else:
                # 不匹配
                print('error---->load pred_model mismatch:' + name)
        
        model_dict.update(new_state_dict)
        cp_model.load_state_dict(model_dict)
        # 保存
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        torch.save(cp_model, ckpt_path)
       