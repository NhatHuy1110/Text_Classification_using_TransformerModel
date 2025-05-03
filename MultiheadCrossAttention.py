import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[[-0.17, -0.29, -0.61],
                   [ 0.48, -1.10, -0.38]]])

c = torch.tensor([[[-0.67, -0.2996, -0.6140],
                   [ 0.52, 0.95, -0.58]]])

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

output_tensor, attn_output_weights = layer(c, x, x)
print(output_tensor)