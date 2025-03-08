import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from dataclasses import dataclass

class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        rms = torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True))
        hidden_states = self.weight * (hidden_states / (rms + self.variance_epsilon))
        
        return hidden_states.to(input_dtype)

class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 较小索引位置对应较低频率
        # 较大的索引位置有较高的频率
        
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

@dataclass
class DeepseekConfig:
    hidden_dim: int = 7168
    num_head: int = 16
    max_position_embeddings: int = 1024  # rope 相关
    rope_theta: int = 128000    # 控制频率
    
    attention_dropout: float = 0.1
    
    q_lora_rank: int = 1536   # latent var shape
    qk_rope_head_dim: int = 64  # 64
    kv_lora_rank: int = 512  # kv latent shape
    
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    
    attention_bias: bool = False
    
class MLA(nn.Module):
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self.config = config
        # part 1. mha
        self.output = nn.Linear(self.config.num_head * self.config.v_head_dim, 
                                self.config.hidden_dim, bias=self.config.attention_bias)
        # part 2. MAL compress
        # down proj
        self.q_down_proj = nn.Linear(self.config.hidden_dim, 
                                     self.config.q_lora_rank, bias=self.config.attention_bias)
        self.q_down_norm = DeepseekV2RMSNorm(config.q_lora_rank)
        self.kv_down_proj = nn.Linear(self.config.hidden_dim,
                                      self.config.kv_lora_rank + config.qk_rope_head_dim,
                                      bias=self.config.attention_bias)
        self.kv_down_norm = DeepseekV2RMSNorm(config.kv_lora_rank)
        
        # up proj
        self.q_up_proj = nn.Linear(config.q_lora_rank,
                                   config.num_head * (config.qk_nope_head_dim + config.qk_rope_head_dim),
                                   bias=config.attention_bias)
        self.kv_up_proj = nn.Linear(config.kv_lora_rank,
                                    config.num_head * (config.qk_nope_head_dim + config.v_head_dim),
                                    bias=config.attention_bias)
        
        # part 3. RoPE
        self.rotary_emb = DeepseekV2RotaryEmbedding(
            config.qk_rope_head_dim,
            config.max_position_embeddings,
            config.rope_theta
        )
    
    def forward(self, hidden_states, position_ids, attn_mask=None):
        # hidden_states shape (bs, seq_len, hidden_dim)
        batch_size, seq_len, _ = hidden_states.size()
        
        # 1. compression
        q = self.q_up_proj(
            self.q_down_norm(
                self.q_down_proj(hidden_states)
            )
        )# q shape (bs, seq_len, num_head*(qk_nope_head_dim + qk_rope_head_dim))
        
        # --> (bs, num_head, seq_len, qk_nope_head_dim + qk_rope_head_dim)
        q = q.view(batch_size, seq_len, self.config.num_head, -1).transpose(1,2)
        
        q_nope, q_rope = torch.split(
            q,
            [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim], 
            dim=-1
        )
        
        # kv_down shape is (bs, seq_len, kv_lora_rank + qk_rope_head_dim)
        kv_down = self.kv_down_proj(hidden_states)

        k_rope, kv_hidden = torch.split(
            kv_down,
            [self.config.qk_rope_head_dim, self.config.kv_lora_rank],
            dim=-1
        )
        k_rope = k_rope.view(batch_size, 1, seq_len, -1)
        
        # kv_up shape is (bs, seq_len, num_head*(qk_nope_head_dim + v_head_dim))
        kv_up = self.kv_up_proj(
            self.kv_down_norm(kv_hidden)
        )
        
        # --> (bs, num_head, seq_len, (qk_nope_head_dim, v_head_dim))
        kv_up = kv_up.view(batch_size, seq_len, self.config.num_head, -1).transpose(1,2)
        k_nope, V = torch.split(
            kv_up,
            [self.config.qk_nope_head_dim, self.config.v_head_dim],
            dim=-1
        )
        
        # apply RoPE
        cos, sin = self.rotary_emb(V, seq_len)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position_ids)
        
        # MHA
        # Q shape is (bs, num_head, seq_len, qk_nope_head_dim + qk_rope_head_dim)
        Q = torch.concat(
            [q_nope, q_rope],
            dim=-1
        )
        K = torch.concat(
            [k_nope, k_rope.expand(-1,self.config.num_head,-1,-1)],
            dim=-1
        )
        attn_weight = (Q @ K.transpose(-1, -2)) / math.sqrt(Q.size(-1))
        if attn_mask is None:
            attn_mask = torch.ones_like(attn_weight)
        attn_mask = torch.tril(attn_mask)
        attn_weight = attn_weight.masked_fill(attn_mask==0, float('-inf'))
        
        attn_weight = F.softmax(attn_weight, dim=-1)
        
        attn_weight = F.dropout(attn_weight, self.config.attention_dropout, training=self.training)
        
        # output shape is (bs, num_head, seq_len, v_head_dim)
        output = attn_weight @ V
        # (bs, seq_len, num_head * v_head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        output = self.output(output)
        
        return output, attn_weight
    
if __name__ == '__main__':
    deepseekConfig = DeepseekConfig()
    bs = 3
    seq_len = deepseekConfig.max_position_embeddings
    mla = MLA(deepseekConfig)
    x = torch.randn(bs, seq_len, deepseekConfig.hidden_dim)
    position_ids = torch.arange(
        deepseekConfig.max_position_embeddings,
    ).unsqueeze(0).expand(
        x.size(0), -1
    ) # (batch_size, seq_len)
    print(mla(x, position_ids)[0].shape, mla(x, position_ids)[1].shape)
    