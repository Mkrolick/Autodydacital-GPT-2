from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layers: int = 6
    n_heads: int = 6
    n_embd: int = 364

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


