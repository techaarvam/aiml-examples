import torch
import numpy as np
from debug import *
from argParser import *
from torch import nn
import common

class Attention(nn.Module):
    def __init__(self):
        super().__init__()   
        # Shape Notes:
          # added num_heads to avoid for loops
          # Input shape will be N - batch_size , num_heads, window_size , vecDim (101)
          # Input shape will be N - batch_size , num_heads, window_size, vecDim 

        # Using guassian init.  
          # TBD: Experiment with scaling factors/distributions later 

          # Note: keys, query, value are the W_k, W_q, W_v
        head_dim = common.vecDims // args.num_heads
        scale = 1.0 / (head_dim ** 0.5)
        self.keys  = nn.Parameter(torch.randn (args.num_heads, common.vecDims, head_dim) * scale)
        self.query = nn.Parameter(torch.randn (args.num_heads, common.vecDims, head_dim) * scale)
        self.value = nn.Parameter(torch.randn (args.num_heads, common.vecDims, head_dim) * scale)

        mask = torch.triu(torch.ones(args.window_size-1, args.window_size-1), diagonal=1).bool()
        self.register_buffer('mask', mask)

    # Shape Notes:
        # X is of shape batch_size, num_heads, window_size, vecDim
        # X is already unsqueezed and num_heads expanded before we get it
    def forward(self, X):
        Q = X @ self.query
        K = X @ self.keys
        V = X @ self.value

        # Shape Notes: (#currently no optimizations applied, its full size KV matrices)
          # TBD for later : downsized V can be used in the Attention and upsized Output matrix in Multi-Head
          

          #W_q(self.query) is num_heads, vecDim, vecDim (look at it as transforming each token into a new linear space)
          # W_k(self.key) is also num_heads, vecDim, vecDim  (look at it as transforming each token into a new linear space)
          # W_v(self.value) is also num_heads, vecDim, vecDim

          # Q @ K.T becomes batch_size, num_heads, window_size, window_size ( all pair of token's get an attention score)
             # pytorch uses colom-vectors. math uses row-vectors. 
             # SoftMax is along dim=-1.  SoftMax answers about the query - i.e which token is the highest for this query.
          # V is batch, num_heads, window_size, vecDim

        # Nuance: pytorch matches last two dimensions. 
          # batch_size, num_heads, window_size, window_size @ num_heads, window_size, vecDim works in pytorch, not in maths!

        head_dim = self.keys.shape[-1]
        scores =  Q @ K.transpose(-2,-1) / (head_dim ** 0.5)
        cur_len = scores.shape[-1]
        if cur_len == self.mask.shape[0]:
            scores.masked_fill_(self.mask, float('-inf'))
        else:
            # inference with context length different from training window — build mask on the fly
            dynamic_mask = torch.triu(torch.ones(cur_len, cur_len, device=scores.device), diagonal=1).bool()
            scores.masked_fill_(dynamic_mask, float('-inf'))
 
        return nn.functional.softmax( scores , dim=-1)  @ V
    


