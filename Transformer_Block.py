import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
        )
        self.ffn = nn.Linear(
            in_features=embed_dim,
            out_features=ff_dim
        )
        self.layernorm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, query, key, value):
        attn_output = self.attn(query, key, value)
        out_1 = self.layernorm_1(query + attn_output)   # residual connection
        ffn_output = self.ffn(out_1)
        out_2 = self.layernorm_2(query + ffn_output)    # residual connection
        return out_2
    
x = torch.tensor(
    [
        [
            [-0.1, 0.1, 0.3],
            [0.4, -1.1, -0.3]
        ]
    ]
)    

embed_dim = 3
ff_dim = 3

# MultiheadAttention
attn = nn.MultiheadAttention(embed_dim, num_heads=1, bias=False, batch_first=True)

custom_weights = torch.tensor( [[-0.3561,  0.3674, -0.5108],
                                [ 0.5146, -0.4764, -0.1490],
                                [ 0.5072, -0.2932, -0.5633],
                                [-0.4932, -0.4468,  0.0736],
                                [-0.6879, -0.4689, -0.1026],
                                [ 0.1847,  0.1858,  0.4469],
                                [-0.4110, -0.4083, -0.5549],
                                [ 0.3921, -0.0746, -0.1336],
                                [-0.6555, -0.3418, -0.2980]]).float()

attn.in_proj_weight = nn.Parameter(custom_weights)

custom_out_proj = torch.tensor([[-0.3601,  0.2771, -0.0573],
                                [-0.0896,  0.0567, -0.2882],
                                [ 0.3200,  0.1517,  0.0580]]).float()

attn.out_proj.weight = nn.Parameter(custom_out_proj)
# MLP
ffn = nn.Linear(in_features=embed_dim, out_features=ff_dim, bias=False)

ffn_weight = torch.tensor([[ 0.1580, -0.4134,  0.5055],
                            [ 0.3910,  0.5469, -0.0767],
                            [-0.3405,  0.4931, -0.4169]]).float()

ffn.weight = nn.Parameter(ffn_weight)

# LayerNorm

layernorm_1 = nn.LayerNorm(normalized_shape=3)
layernorm_2 = nn.LayerNorm(normalized_shape=3)

# Computation
query=x
key=x
value=x

attn_output, _ = attn(query, key, value)
out_1 = layernorm_1(query+attn_output)
ffn_output = ffn(out_1)
out_2 = layernorm_2(query+ffn_output)

print("Output: ", out_2)