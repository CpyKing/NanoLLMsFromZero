import torch
import torch.nn as nn
import math

class GroupQueryAttn(nn.Module):
    def __init__(self, hidden_dim, num_head, num_group):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.head_dim = hidden_dim // num_head
        self.num_group = num_group
        self.num_kv_head = self.num_head // self.num_group
        self.kv_head_dim = self.num_kv_head * self.head_dim
        
        self.q_proj = nn.Linear(hidden_dim, self.head_dim * self.num_head)
        self.k_proj = nn.Linear(hidden_dim, self.head_dim * self.num_kv_head)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim * self.num_kv_head)
        
        # dropout ...
        self.attn_output = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, X, attn_mask=None):
        batch_size, seq_len, _ = X.size()
        
        # X shape is (bs, seq, hidden_dim)
        # Q shape is (bs, seq, hidden_dim = num_head * head_dim)
        # K V shape is (bs, seq, kv_head_dim = num_kv_head * head_dim)
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        # Q shape is (bs, num_head, seq, head_dim)
        # K V shape is (bs, num_kv_head, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_head, -1).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_head, -1).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_kv_head, -1).transpose(1,2)
        
        # K V shape is (bs, num_head, seq, head_dim)
        K = K.repeat_interleave(self.num_group, dim=1)
        V = V.repeat_interleave(self.num_group, dim=1)
        
        
        
        attn_weight = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # TODO attn mask ...
        attn_weight = torch.softmax(attn_weight, dim=-1)
        # TODO dropout
        h = attn_weight @ V
        
        h = h.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        h = self.attn_output(h)
        return h


if __name__ == '__main__':
    batch_size = 3
    seq_len = 4
    hidden_dim = 64
    num_head = 8
    num_group = 2
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    print(x.shape)
    gqa = GroupQueryAttn(hidden_dim, num_head, num_group)
    print(gqa(x).shape)
        
        