import torch
import math
import torch.nn as nn

class SelfAttnV1(nn.Module):
    def __init__(self, hidden_dim: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, X):
        # X shape is (batch_size, seq_len, dim)
        # Q K V shape is (batch_size, seq_len, dim)
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        # attn_weight shape is (batch_size, seq_len, seq_len)
        attn_weight = torch.softmax(
            (Q @ K.transpose(-1, -2)) / math.sqrt(self.hidden_dim),
            dim=-1
        )
        
        h = attn_weight @ V
        
        return h
    
class SelfAttnV2(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
    
    def forward(self, X):
        qkv = self.qkv_proj(X)
        
        Q, K, V = torch.split(qkv, self.hidden_dim, dim=-1)
        attn_weight = torch.softmax(
            (Q @ K.transpose(-1, -2)) / math.sqrt(self.hidden_dim),
            dim=-1
        )
        
        h = attn_weight @ V
        return h
    
class SelfAttnV3(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, X, attn_mask=None):
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        attn_weight = (Q @ K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        if attn_mask is not None:
            attn_weight = attn_weight.masked_fill(
                attn_mask == 0,
                float("-inf")
            )
        attn_weight = torch.softmax(attn_weight, dim=-1)
        
        attn_weight = self.dropout(attn_weight)
        
        h = attn_weight @ V
        h = self.output_layer(h)
        return h
    
class MultiHeadAttn(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim  = hidden_dim // head_num
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout_layer = nn.Dropout(dropout_rate)
        
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, X, attn_mask=None):
        batch_size, seq_len, _ = X.size()
        
        # Q K V shape is (batch_size, seq_len, hidden_dim)
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        # (batch_size, seq_len, hidden_dim) --> (batch_size, head_num, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.head_num, -1).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, -1).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, -1).transpose(1, 2)
        
        attn_weight = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_weight = attn_weight.masked_fill(
                attn_mask == 0,
                float("-inf")
            )
        attn_weight = torch.softmax(attn_weight, dim=-1)
        
        attn_weight = self.dropout_layer(attn_weight)
        
        h = attn_weight @ V
        
        # (batch_size, head_num, seq_len, head_dim) --> (batch_size, seq_len, hidden_dim)
        h = h.transpose(1, 2).contiguous()
        h = h.view(batch_size, seq_len, -1)
        
        h = self.output_layer(h)
        return h
    
if __name__ == '__main__':
    batch_size = 3
    seq_len = 4
    hidden_dim = 9
    head_num = 3
    self_attn_v1 = SelfAttnV1(hidden_dim = hidden_dim)
    self_attn_v2 = SelfAttnV2(hidden_dim = hidden_dim)
    self_attn_v3 = SelfAttnV3(hidden_dim = hidden_dim, dropout_rate=0.2)
    multi_head_attn = MultiHeadAttn(hidden_dim=hidden_dim, head_num=head_num, dropout_rate=0.2)
    
    X = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.tensor(
        [
            [1,1,1,0],
            [1,0,0,0],
            [1,1,1,1]
        ]
    )
    mask = mask.unsqueeze(dim = 1).repeat(1,seq_len,1)
    
    print("X shape is:", X.shape)
    
    print(self_attn_v1(X).shape)
    print(self_attn_v2(X).shape)
    print(self_attn_v3(X, mask).shape)
    print(multi_head_attn(X, mask).shape)
    
    