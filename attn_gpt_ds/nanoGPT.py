import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

torch.manual_seed(1024)

@dataclass
class GPTConfig:
    block_size: int = 512    # max_seq
    batch_size: int = 16
    num_layer: int = 6 #12
    num_head: int = 4  #12
    emb_hidden_dim: int = 384 #768
    dropout: float = 0.1
    head_dim: int =  emb_hidden_dim // num_head
    # vocab_size
    # gpt2 official tokenizer
    vocab_size: int = 50257
    
# 1. single head attention
class SingleHeadAttn(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.hidden_dim = config.emb_hidden_dim
        
        self.q_proj = nn.Linear(config.emb_hidden_dim, config.head_dim)
        self.k_proj = nn.Linear(config.emb_hidden_dim, config.head_dim)
        self.v_proj = nn.Linear(config.emb_hidden_dim, config.head_dim)
        
        self.register_buffer(
            "attn_mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            )
        )
        
        self.drop_out = nn.Dropout(config.dropout)
        
    def forward(self, X):
        batch_size, seq_len, _ = X.size()
        
        # Q K V shape is (bs, seq_len, hidden_dim)
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        # attn_weight shape is (bs, seq_len, seq_len)
        attn_weight = (Q @ K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        attn_weight = attn_weight.masked_fill(
            self.attn_mask[:seq_len, :seq_len] == 0,
            float("-inf")
        )
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.drop_out(attn_weight)
        
        h = attn_weight @ V
        return h

# 2. MultiHeadAttn
class MultiHeadAttn(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_heads = nn.ModuleList(
            [
                SingleHeadAttn(config) for _ in range(config.num_head)
            ]
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.output = nn.Linear(config.emb_hidden_dim, config.emb_hidden_dim)
    
    def forward(self, X):
        h = torch.cat(
            [l(X) for l in self.attn_heads], dim=-1
        )
        h = self.output(h)
        h = self.dropout(h)
        return h

## 3. feed forward layer
class FFN(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net =  nn.Sequential(
            nn.Linear(config.emb_hidden_dim, 4 * config.emb_hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.emb_hidden_dim, config.emb_hidden_dim),   
            nn.Dropout(config.dropout)
        )
        
    def forward(self, X):
        return self.net(X)
    
## 4. block
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_0 = nn.LayerNorm(config.emb_hidden_dim, eps=1e-5)
        self.mha = MultiHeadAttn(config)
        self.ln_1 = nn.LayerNorm(config.emb_hidden_dim, eps=1e-5)
        self.ffn = FFN(config)
    def forward(self, X):
        ln_0_res = self.ln_0(X)
        mha_res = self.mha(ln_0_res)
        mha_res = mha_res + X
        
        ln_1_res = self.ln_1(mha_res)
        ffn_res = self.ffn(ln_1_res)
        ffn_res = ffn_res + mha_res
        
        return ffn_res
    
## 5. GPT
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # embedding position, block * 12, norm mlp softmax
        # position embedding: 0, 1, ... embedding --upgrade--> rope
        # norm: LayerNorm --upgrade--> RMSNorm
        # mlp: --upgrade--> swiglu
        # mha: --upgrade--> gqa
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_hidden_dim)
        self.pos_emb = nn.Embedding(config.block_size, config.emb_hidden_dim)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layer)])
        self.ln_final = nn.LayerNorm(config.emb_hidden_dim, eps=1e-5)
        self.lm_head = nn.Linear(config.emb_hidden_dim, config.vocab_size, bias=False)
        # 现在的 SLM 模型 会用 tie weight 来减少参数
        # ** 学习一下 tie weight 以及为什么两个层的参数可以绑定 **
        self.tok_emb.weight = self.lm_head.weight
        
    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            # 初始化为正态分布
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 初始化为正态分布
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        # idx 输入的 token ids
        # targets 输出的 tokenids
        # shape 要一样
        batch_size, seq_len = idx.size()
        
        tok_emb_res = self.tok_emb(idx) # (bs, seq_len, hidden_dim)
        pos_emb_res = self.pos_emb(torch.arange(seq_len, device=tok_emb_res.device))
        block_input = tok_emb_res + pos_emb_res
        
        block_output = self.blocks(block_input)
        
        ln_final_output = self.ln_final(block_output)
        logits = self.lm_head(ln_final_output)
        # logits = torch.softmax(lm_head_output, dim=-1)
        
        
        # compute loss
        if targets is None:
            loss = None
        else:
            batch_size, seq_len, voc_size = logits.size()
            logits = logits.view(batch_size * seq_len, voc_size)
            targets = targets.view(batch_size * seq_len)
            
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx (bs, T)
        for _ in range(max_new_tokens):
            idx_temp = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_temp)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx