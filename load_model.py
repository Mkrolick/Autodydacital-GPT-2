from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import tiktoken
import argparse

# ---
# GPT Definitions:


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50527
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config : GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # k,q,x projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOSCALE_GPT_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1,config.block_size,config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C // self.n_head).transpose(1,2)

        # Basic attension
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        # Flash attension
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config : GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config : GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config :GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        # init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOSCALE_GPT_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of size {T} as blocksize is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets != None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load models from the huggingface hub"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=38, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # initialize huggingface model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes 
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over other parameter
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict =  {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim options. Any paramter that is 2D will be weight destroyed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed params tensor: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed params tensor: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == 'cuda'
        print(f"used fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused=use_fused)
        return optimizer


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')  # Fall back to CPU instead of MPS

def load_model(checkpoint_path):
    # Determine device
    device = get_default_device()
    
    # Load the checkpoint with appropriate device mapping
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    
    # Create a new model instance with the saved config
    model = GPT(checkpoint['config'])
    
    # Load the saved model state
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    
    # Create optimizer if you plan to continue training
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
    
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        optimizer = None
    
    # Set to eval mode if doing inference
    model.eval()
    
    return model, optimizer, checkpoint['step']

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate text using GPT model')
    parser.add_argument('--prompt', type=str, default="Hello, I'm a language model,",
                       help='Input prompt for text generation')
    parser.add_argument('--model', type=str, default="ckpt.pt",
                       help='Model specified for generation')
    args = parser.parse_args()

    # Just load the model without trying to create a DataLoader
    model, optimizer, step = load_model('log/' + args.model)
    print(f"Successfully loaded model at step {step}")

    device = get_default_device()

    model.eval()
    num_return_sequences = 1

    start_len = len(args.prompt)
    max_len = 128
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(args.prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

    # inline generated sequence
    x = tokens.to(device)
    # create a generator object inline to not interfere with global generator object
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)
    while x.size(1) - start_len < max_len:
        # forward model to get logits
        with torch.no_grad():
            #with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, _ = model(x) # (B, T, vocab_size)
            # take logits at last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indivies is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from top-k probabilities
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to sequence
            x = torch.cat((x,xcol), dim=1)

    # print generated text
    for i in range(num_return_sequences):
        tokens = x[i, start_len:max_len+start_len].tolist()
        decoded = enc.decode(tokens)
        print(f"decoded: {args.prompt + decoded}")

