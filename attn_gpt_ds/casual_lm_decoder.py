import torch
import torch.nn as nn
import math

class MultiHeadAttn(nn.Module):
    def __init__(self, head_num, hidden_dim, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.attn_dropout = nn.Dropout(dropout_rate)
        
        self.attn_output = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, X, mask):
        batch_size, seq_len, _ = X.size()
        
        # X shape is (bs, seq, hid_dim)
        # Q K V shape is (bs, seq, hid_dim)
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        # (bs, seq, hid_dim) --> (bs, head_num, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.head_num, -1).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, -1).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, -1).transpose(1, 2)
        
        # get attn weight
        # attn_weight shape is (bs, head_num, seq, seq)
        attn_weight = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is None:
            mask = torch.ones_like(attn_weight)
        mask = mask.tril()  # casual lm mask
        attn_weight = attn_weight.masked_fill(mask==0, float('-inf'))
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.attn_dropout(attn_weight)
        
        # h shape is (bs, head_num, seq, head_dim)
        h = attn_weight @ V
        
        # (bs, head_num, seq, head_dim) --> (bs, seq, hid_dim)
        h = h.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        h = self.attn_output(h)
        
        return h

class FFN(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.up_layer = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act_fn = nn.ReLU()
        self.down_layer = nn.Linear(hidden_dim * 4, hidden_dim)
        
        self.ffn_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, X):
        up_res = self.up_layer(X)
        up_res = self.act_fn(up_res)
        
        down_res = self.down_layer(up_res)        
        
        down_res = self.ffn_dropout(down_res)
        
        return down_res

class DecoderLayer(nn.Module):
    def __init__(self, head_num, hidden_dim, attn_dropout_rate=0.2, ffn_dropout_rate=0.2):
        super().__init__()
        self.mha = MultiHeadAttn(head_num=head_num, hidden_dim=hidden_dim, dropout_rate=attn_dropout_rate)
        self.mha_ln = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.ffn = FFN(hidden_dim=hidden_dim, dropout_rate=ffn_dropout_rate)
        self.ffn_ln = nn.LayerNorm(hidden_dim, eps=1e-5)
        
    def forward(self, X, mask):
        # multiheadattn + post norm
        mha_res = self.mha(X, mask)
        mha_res = self.mha_ln(X + mha_res)
        
        # ffn + post norm
        ffn_res = self.ffn(mha_res)
        ffn_res = self.ffn_ln(mha_res + ffn_res)
        
        return ffn_res
    
class Decoder(nn.Module):
    def __init__(self, num_dec_layers, emb_input, 
                 head_num, hidden_dim, attn_dropout_rate=0.2, ffn_dropout_rate=0.2):
        super().__init__()
        self.emb = nn.Embedding(emb_input, hidden_dim)
        self.dec_layers = nn.ModuleList(
            [
                DecoderLayer(head_num, hidden_dim, attn_dropout_rate, ffn_dropout_rate) for i in range(num_dec_layers)
            ]
        )
        self.output = nn.Linear(hidden_dim, emb_input)
    
    def forward(self, X, mask):
        X = self.emb(X)
        
        
        for dec_layer in self.dec_layers:
            X = dec_layer(X, mask)
        
        output = self.output(X)
        output = torch.softmax(output, dim=-1)
        
        return output
    
if __name__ == '__main__':
    batch_size = 3
    head_num = 8
    seq_len = 4
    emb_input = 32
    num_dec_layers = 5
    hidden_dim = 64
    
    X = torch.randint(low=0, high=emb_input, size=(batch_size, seq_len))
    my_decoder = Decoder(num_dec_layers, emb_input, head_num,
                         hidden_dim)
    print(X.shape)
    mask = torch.tensor(
        [
            [1,1,1,0],
            [1,0,0,0],
            [1,1,1,1]
        ]
    )
    mask = mask.unsqueeze(dim = 1).unsqueeze(dim = 2).repeat(1,head_num, seq_len,1)
    print(my_decoder(X, mask).shape)