from abc import ABC, abstractmethod
from typing import Dict, List, Callable
import torch
from torch import Tensor


class BaseProcessor(ABC):
    """处理器抽象基类 - 策略模式"""
    
    @abstractmethod
    def process(self, input_dict: dict) -> dict:
        """处理单个数据项"""
        pass
    
    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        """输出字段列表"""
        pass


class PreTrainProcessor(BaseProcessor):
    """预训练数据处理器"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def process(self, input_dict: dict) -> dict:
        segment = input_dict["text"]
        tokens = self.tokenizer.encode(f"{segment}<eos>")
        return {'sequence': torch.tensor(tokens, dtype=torch.int32)}
    
    @property
    def output_keys(self) -> List[str]:
        return ["sequence"]


class SFTProcessor(BaseProcessor):
    """监督微调数据处理器"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def process(self, input_dict: dict) -> dict:
        query, response = input_dict["query"], input_dict["response"]
        q = self.tokenizer.encode(
            f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        )
        a = self.tokenizer.encode(f"{response}<|im_end|>\n<eos>")
        
        tokens = torch.tensor(q + a, dtype=torch.int32)
        loss_mask = torch.zeros_like(tokens, dtype=torch.bool)
        loss_mask[len(q):] = True
        
        return {"sequence": tokens, "loss_mask": loss_mask}
    
    @property
    def output_keys(self) -> List[str]:
        return ["sequence", "loss_mask"]


class DPOProcessor(BaseProcessor):
    """DPO偏好学习数据处理器"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def process(self, input_dict: dict) -> dict:
        # TODO: 实现DPO处理逻辑
        return None
    
    @property
    def output_keys(self) -> List[str]:
        return ["chosen", "chosen_mask", "rejected", "rejected_mask"]


class ProcessorFactory:
    """处理器工厂 - 工厂模式"""
    
    _processors = {
        "pt": PreTrainProcessor,
        "sft": SFTProcessor,
        "dpo": DPOProcessor,
    }
    
    @classmethod
    def create(cls, processor_type: str, tokenizer) -> BaseProcessor:
        """创建处理器实例"""
        if processor_type not in cls._processors:
            raise ValueError(f"Invalid processor type: {processor_type}")
        return cls._processors[processor_type](tokenizer)
    
    @classmethod
    def register(cls, processor_type: str, processor_class: type):
        """注册新的处理器类型"""
        cls._processors[processor_type] = processor_class


# 向后兼容的函数接口
def get_pt_processor(tokenizer) -> Callable:
    processor = PreTrainProcessor(tokenizer)
    return processor.process

def get_sft_processor(tokenizer) -> Callable:
    processor = SFTProcessor(tokenizer)
    return processor.process

def get_dpo_processor(tokenizer) -> Callable:
    processor = DPOProcessor(tokenizer)
    return processor.process
