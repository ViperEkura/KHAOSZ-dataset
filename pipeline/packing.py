from typing import List
import torch
from torch import Tensor


class SequencePacker:
    """序列打包（bin-packing）"""

    def __init__(self, pack_size: int, pad_value: int = 0):
        self.pack_size = pack_size
        self.pad_value = pad_value

    def pack(self, sequences: List[Tensor]) -> List[Tensor]:
        packages = []
        sequences.sort(key=lambda x: x.numel(), reverse=True)

        current_pack = torch.full((self.pack_size,), self.pad_value, dtype=torch.int32)
        current_pos = 0

        for tensor in sequences:
            tensor = tensor[:self.pack_size] if tensor.numel() > self.pack_size else tensor
            tensor_size = tensor.numel()

            if current_pos + tensor_size > self.pack_size:
                packages.append(current_pack)
                current_pack = torch.full((self.pack_size,), self.pad_value, dtype=torch.int32)
                current_pos = 0

            current_pack[current_pos:current_pos + tensor_size] = tensor
            current_pos += tensor_size

        if current_pos > 0:
            packages.append(current_pack)

        return packages
