import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from basicMOE import BasicExpert

@dataclass
class MOEConfig:
    hidden_dim: int = 32
    num_experts: int = 7
    top_k: int = 3
    num_shared_experts: int = 2

class MOERouter(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_dim, config.num_experts)
    
    def forward(self, X):
        # router_logits shape is (bs * seq_len, num_experts)
        router_logits = self.gate(X)
        
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        # topk 可以反向传播
        # (bs * seq_len, top_k)
        router_weights, selected_experts_indices = torch.topk(
            router_probs,
            self.config.top_k,
            dim=-1
        )
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(X.dtype)
        
        expert_mask = F.one_hot(selected_experts_indices, self.config.num_experts)  # (bs * seq_len, top_k, num_experts)
        
        expert_mask = expert_mask.permute(2,1,0) # (num_experts, top_k, bs * seq_len)
        return router_logits, router_weights, selected_experts_indices, expert_mask
        
class SpaseMOE(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()
        self.config = MOEConfig
        
        self.expert_list = nn.ModuleList(
            [
                BasicExpert(self.config.hidden_dim, self.config.hidden_dim) for _ in range(self.config.num_experts)
            ] 
        )
        self.router = MOERouter(self.config)
        
    def forward(self, X):
        batch_size, seq_len, _ = X.size()
        
        # token 纬度计算 X reshape --> (bs * seq_len, hidden_dim)
        X = X.view(-1, self.config.hidden_dim)
        
        # expert comp
        router_logits, router_weight, selected_experts_indices, expert_mask = self.router(X)
        # expert_mask shape is (num_experts, top_k, bs * seq_len)
        
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, self.config.hidden_dim),
            dtype=X.dtype,
            device=X.device
        )
        
        for expert_idx in range(self.config.num_experts):
            expert_layer = self.expert_list[expert_idx]
            router_idx, token_idx = torch.where(expert_mask[expert_idx])
            import pdb;pdb.set_trace()
            
            # 取到 token_idx 对应的 X
            # (len(token_idx), hidden_dim)
            current_state = X.unsqueeze(0)[:, token_idx, :].reshape(-1, self.config.hidden_dim)
            
            current_state_res = expert_layer(current_state) * router_weight[token_idx, router_idx].unsqueeze(-1)
        
            final_hidden_states.index_add_(0, token_idx, current_state_res.to(X.dtype))
        
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, -1)
        return final_hidden_states, router_logits
    
class SharedExpertMOE(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()
        self.config = config
        self.sparseMOE = SpaseMOE(config)
        self.shared_expert_list = nn.ModuleList(
            [
                BasicExpert(self.config.hidden_dim, self.config.hidden_dim) for _ in range(self.config.num_shared_experts)
            ]
        )
    
    def forward(self, X):
        batch_size, seq_len, _ = X.size()
        
        shared_expert_output_list = [expert(X) for expert in self.shared_expert_list]
        shared_expert_output_list = torch.stack(shared_expert_output_list, dim=0) # (num_shared_expert, bs, seq_len, hidden_dim)
        
        shared_expert_out = shared_expert_output_list.sum(dim=0)    # (bs, seq_len, hidden_dim)
        sparsemoe_out, router_logits = self.sparseMOE(X)
        output = shared_expert_out + sparsemoe_out
        return output, router_logits
        
        

if __name__ == '__main__':
    batch_size = 64
    seq_len = 512
    moeconfig = MOEConfig()
    x = torch.randn(batch_size, seq_len, moeconfig.hidden_dim)
    sparsemoe = SpaseMOE(moeconfig)
    sharedmoe = SharedExpertMOE(moeconfig)
    print(sparsemoe(x)[0].shape)
    print(sharedmoe(x)[0].shape)
        