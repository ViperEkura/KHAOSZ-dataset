from pathlib import Path
from typing import Dict, List, Union
import os
import h5py
import torch
from torch import Tensor


class IOHandler:
    """文件和H5数据存储处理器"""
    
    @staticmethod
    def fetch_files(directory: str) -> List[str]:
        """获取目录下所有文件"""
        return [
            os.path.join(root, f) 
            for root, _, files in os.walk(directory) 
            for f in files
        ]
    
    @staticmethod
    def fetch_folders(root_dir: str, filter_func=None) -> List[str]:
        """获取目录下所有文件夹"""
        folders = []
        for root, dirs, _ in os.walk(root_dir):
            for dir_name in dirs:
                folder_path = os.path.join(root, dir_name)
                if filter_func is None or filter_func(folder_path):
                    folders.append(folder_path)
        return folders
    
    @staticmethod
    def save_h5(file_path: str, file_name: str, tensor_group: Dict[str, List[Tensor]]) -> None:
        """保存张量组到H5文件"""
        os.makedirs(file_path, exist_ok=True)
        full_path = os.path.join(file_path, f"{file_name}.h5")
        
        with h5py.File(full_path, 'w') as f:
            for key, tensors in tensor_group.items():
                grp = f.create_group(key)
                for idx, tensor in enumerate(tensors):
                    grp.create_dataset(f'data_{idx}', data=tensor.cpu().numpy())
    
    @staticmethod
    def load_h5(file_path: str, share_memory: bool = True) -> Dict[str, List[Tensor]]:
        """从H5文件加载张量组"""
        tensor_group: Dict[str, List[Tensor]] = {}
        root_path = Path(file_path)
        h5_files = list(root_path.rglob("*.h5")) + list(root_path.rglob("*.hdf5"))
        
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                for key in f.keys():
                    grp = f[key]
                    tensors = [
                        (torch.from_numpy(dset[:]).share_memory_() if share_memory 
                         else torch.from_numpy(dset[:]))
                        for dset_name in grp.keys()
                        for dset in [grp[dset_name]]
                    ]
                    tensor_group.setdefault(key, []).extend(tensors)
        
        return tensor_group


# 向后兼容的函数接口
def fetch_files(directory: str) -> List[str]:
    return IOHandler.fetch_files(directory)

def fetch_folders(root_dir: str, filter_func=None) -> List[str]:
    return IOHandler.fetch_folders(root_dir, filter_func)

def save_h5(file_path: str, file_name: str, tensor_group: Dict[str, List[Tensor]]) -> None:
    return IOHandler.save_h5(file_path, file_name, tensor_group)

def load_h5(file_path: str, share_memory: bool = True) -> Dict[str, List[Tensor]]:
    return IOHandler.load_h5(file_path, share_memory)
