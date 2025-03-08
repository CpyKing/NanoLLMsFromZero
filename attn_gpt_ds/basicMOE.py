import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicExpert(nn.Module):
    def __init__(self, feat_in, feat_out):
        super().__init__()
        self.fc = nn.Linear(feat_in, feat_out)
        
    def forward(self, X):
        return self.fc(X)


class BasicMOE(nn.Module):
    def __init__(self, feat_in, feat_out, num_experts):
        super().__init__()
        self.gate = nn.Linear(feat_in, num_experts)
        self.expert_list = nn.ModuleList(
            [BasicExpert(feat_in, feat_out) for _ in range(num_experts)]
        )
    
    def forward(self, X):
        # X shape is (bs, feat_in) or (bs, hidden_dim)
        # expert_weight shape is (bs, num_experts)
        expert_weight = self.gate(X)
        expert_weight = F.softmax(expert_weight, dim=-1)
        # expert(x) shape is (bs, feat_out)
        expert_out_list = [expert(X) for expert in self.expert_list]
        
        # expert_output shape is (bs, num_experts, feat_out)
        expert_output = torch.concat(
            [
                expert_out.unsqueeze(dim=1) for expert_out in expert_out_list
            ],
            dim=1
        )
        
        # output shape is (bs, 1, feat_out)
        expert_weight = torch.unsqueeze(expert_weight, dim=1)
        output = expert_weight @ expert_output
        
        return output.squeeze(dim=1)
        
        
if __name__ == '__main__':
    batch_size = 3
    feat_in = 5
    feat_out = 4
    num_experts = 6
    
    x = torch.randn(batch_size, feat_in)
    basic_moe = BasicMOE(feat_in, feat_out, num_experts)
    
    print(basic_moe(x).shape)