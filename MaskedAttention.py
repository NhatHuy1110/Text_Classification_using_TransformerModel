import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[[-0.1, 0.1, 0.3],
                   [0.4, -1.1, -0.3]]])

layer = nn.MultiheadAttention(embed_dim=3, num_heads=1, bias=False, batch_first=True)

custom_weights = torch.tensor( [[-0.3561,  0.3674, -0.5108],
                                [ 0.5146, -0.4764, -0.1490],
                                [ 0.5072, -0.2932, -0.5633],
                                [-0.4932, -0.4468,  0.0736],
                                [-0.6879, -0.4689, -0.1026],
                                [ 0.1847,  0.1858,  0.4469],
                                [-0.4110, -0.4083, -0.5549],
                                [ 0.3921, -0.0746, -0.1336],
                                [-0.6555, -0.3418, -0.2980]]).float()

layer.in_proj_weight = nn.Parameter(custom_weights)

custom_out_proj = torch.tensor([[-0.3601,  0.2771, -0.0573],
                                [-0.0896,  0.0567, -0.2882],
                                [ 0.3200,  0.1517,  0.0580]]).float()

layer.out_proj.weight = nn.Parameter(custom_out_proj)

# Create an upper triangular mask for casual attention
# Adjust the size as per your sequence length
mask = torch.triu(torch.ones(1, 2, 2), diagonal=1).bool()
print(f"mask: {mask}")

output_tensor, attn_output_weights = layer(x, x, x, attn_mask=mask)

print(f"output_tensor: {output_tensor}")


#############################################################################

x = torch.tensor([[[-0.1, 0.1, 0.3],
                   [0.4, -1.1, -0.3]]])

q = torch.tensor(  [[-0.3561,  0.3674, -0.5108],
                    [ 0.5146, -0.4764, -0.1490],
                    [ 0.5072, -0.2932, -0.5633]]).float()
k = torch.tensor(  [[-0.4932, -0.4468,  0.0736],
                    [-0.6879, -0.4689, -0.1026],
                    [ 0.1847,  0.1858,  0.4469]]).float()
v = torch.tensor(  [[-0.4110, -0.4083, -0.5549],
                    [ 0.3921, -0.0746, -0.1336],
                    [-0.6555, -0.3418, -0.2980]]).float()
o = torch.tensor([[-0.3601,  0.2771, -0.0573],
                  [-0.0896,  0.0567, -0.2882],
                  [ 0.3200,  0.1517,  0.0580]]).float()

embed_dim = 3
num_heads = 1
head_dim = embed_dim // num_heads

query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

query_proj.weight = nn.Parameter(q)
key_proj.weight = nn.Parameter(k)
value_proj.weight = nn.Parameter(v)

query = query_proj(x)
key = key_proj(x)
value = value_proj(x)

# (batch_size, num_heads, seq_len, head_dim)
query = query.view(1, num_heads, -1, head_dim)
key = key.view(1, num_heads, -1, head_dim)
value = value.view(1, num_heads, -1, head_dim)

attention_scores = torch.matmul(query, key.transpose(-2, -1))/(head_dim ** 0.5)

seq_len = attention_scores.size(-1)
mask = torch.triu(torch.ones(1, 1, seq_len, seq_len), diagonal=1).bool()

attention_scores = attention_scores.masked_fill(mask, float("-inf"))

attention_weights = F.softmax(attention_scores, dim=-1)
context = torch.matmul(attention_weights, value)

print('attention_weights: ', attention_weights)
print('context: ', context)

output = context.view(1, -1, embed_dim)
output = output@o.T
print(output)