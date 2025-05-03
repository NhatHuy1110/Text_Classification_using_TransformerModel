import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([
    [
        [-0.1, 0.1, 0.3]
    ]
])

layer = nn.MultiheadAttention(embed_dim=3, num_heads=1, bias=False, batch_first=True)

custom_weight = torch.tensor( [[-0.3561,  0.3674, -0.5108],
                                [ 0.5146, -0.4764, -0.1490],
                                [ 0.5072, -0.2932, -0.5633],
                                [-0.4932, -0.4468,  0.0736],
                                [-0.6879, -0.4689, -0.1026],
                                [ 0.1847,  0.1858,  0.4469],
                                [-0.4110, -0.4083, -0.5549],
                                [ 0.3921, -0.0746, -0.1336],
                                [-0.6555, -0.3418, -0.2980]]).float()

layer.in_proj_weight = nn.Parameter(custom_weight)

custom_out_proj = torch.tensor([[-0.3601,  0.2771, -0.0573],
                                [-0.0896,  0.0567, -0.2882],
                                [ 0.3200,  0.1517,  0.0580]]).float()

layer.out_proj.weight = nn.Parameter(custom_out_proj)

output_tensor, attn_output_weight = layer(x, x, x)

# print("output_tensor", output_tensor)

#########################################################################

q = torch.tensor([[-0.3561,  0.3674, -0.5108],
                [ 0.5146, -0.4764, -0.1490],
                [ 0.5072, -0.2932, -0.5633]]).float()
k = torch.tensor([[-0.4932, -0.4468,  0.0736],
                [-0.6879, -0.4689, -0.1026],
                [ 0.1847,  0.1858,  0.4469]]).float()
v = torch.tensor([[-0.4110, -0.4083, -0.5549],
                [ 0.3921, -0.0746, -0.1336],
                [-0.6555, -0.3418, -0.2980]]).float()
o = torch.tensor([[-0.3601,  0.2771, -0.0573],
                [-0.0896,  0.0567, -0.2882],
                [ 0.3200,  0.1517,  0.0580]]).float()

embed_dim = 3
num_heads = 1
head_dim = embed_dim // num_heads

# Step 1: Linear Projections for queries, keys, and values
query_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,bias=False)
key_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,bias=False)
value_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,bias=False)

# Custom weights for linear projections
query_proj.weight = nn.Parameter(q)
key_proj.weight = nn.Parameter(k)
value_proj.weight = nn.Parameter(v)

# Step 2: Split the input into multiple heads
query = query_proj(x)
key = key_proj(x)
value = value_proj(x)

# Reshape query, key, and value to have shape(batch_size, num_heads, seq_len, head_dim)
query = query.view(1, num_heads, -1, head_dim)
key = key.view(1, num_heads, -1, head_dim)
value = value.view(1, num_heads, -1, head_dim)

# Step 3: Compute scaled dot-product attention
attention_output = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
attention_weight = F.softmax(attention_output, dim=-1)
context = torch.matmul(attention_weight, value)

# Step 4: Concatenate and project back
output = context.view(1, -1, embed_dim)
output_proj = nn.Linear(embed_dim, embed_dim, bias=False)
output_proj.weight = nn.Parameter(o)
output = output_proj(output)

print("output:", output)


########################################################################

x = torch.tensor([[[-0.1, 0.1, 0.3]]])
x = x.reshape(1, 3)

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
num_heads = 3
head_dim = embed_dim // num_heads

# Step 1: Linear projections for queries, keys, and values
query = x@q.T
key = x@k.T
value = x@v.T

query = query.view(num_heads, -1, head_dim)
key = key.view(num_heads, -1, head_dim)
value = value.view(num_heads, -1, head_dim)

attention_output = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
attention_weight = F.softmax(attention_output, dim=-1)
context = torch.matmul(attention_weight, value)

output1 = context.view(-1, embed_dim)
output1 = output@o.T
print("output::", output1)