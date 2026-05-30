import torch
import numpy as np
from debug import *
from argParser import *
from torch import nn
import common
import torch.nn.functional as F

#TBD: rename this file and class as MultiHeadAttention
# This file hands all the heads. Its not the attention block for a single-head
# Its the multi-head attention
# rename MultiHead.py to transformer.py also
# The instance created is assigned in trainer.py to a variable named transformer, which
# is better reflective of the naming 

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
        head_dim = common.innerDims // args.num_heads
        scale = 1.0 / (head_dim ** 0.5)
        if (args.qkv == "unfused"):
            self.keys  = nn.Parameter(torch.randn (args.num_heads, common.innerDims, head_dim) * scale)
            self.query = nn.Parameter(torch.randn (args.num_heads, common.innerDims, head_dim) * scale)
            self.value = nn.Parameter(torch.randn (args.num_heads, common.innerDims, head_dim) * scale)
        else: 
            # the fun in linear-algebra is how, adding more rows or columns, keeps the operations
            # often the same and equivalent. instead of num_heads * head_dim
            # qkv has merged them and keeping them together. 
            # This could have been done in the unfused code path also. 
            # but implementing it right in the fused code path first.

            self.qkv =  nn.Parameter( torch.randn (common.innerDims, 3*common.innerDims) * scale)

        mask = torch.triu(torch.ones(args.window_size-1, args.window_size-1), diagonal=1).bool()
        self.register_buffer('mask', mask)

    # Shape Notes:
        # X is of shape batch_size, num_heads, window_size, vecDim
        # X is already unsqueezed and num_heads expanded before we get it
    def forward(self, X):
        # For educational value and measuring perf/ram, preserving the old unfused implementation 
        # and using an argument to select between the two.
        if (args.qkv == "unfused"):
            return self.unfused_forward(X)
        else:
            return self.fused_forward(X)


    def fused_forward(self,X):
        B, W, _ = X.shape
        H = args.num_heads
        head_dim = common.innerDims // H

        qkv = (X @ self.qkv).reshape(B, W, 3, H, head_dim)
        # SDPA expects [B, H, W, head_dim]
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return output.permute(0, 2, 1, 3).reshape(B, W, H*head_dim)

    def unfused_forward(self,X):

        X = X.unsqueeze(1).expand(-1, args.num_heads, -1, common.innerDims)
        #Claude sugggested changing this to use einsum.
     
        # when normedX expansion calls einsim instead of this current code.
        #   X will be read H times, since cuBLAS sees [B S I ] @ [H I D] as a loop 
        #   over H, each using the same X in-place
        #   but cuda compile is called, so the claim of improvement may be close to zero or non-existent. 
        #   expecting cuda compile to take care of forward optimising.

        Q = X @ self.query
        K = X @ self.keys
        V = X @ self.value

        # Shape Notes: (#currently no optimizations applied, its full size KV matrices)
          # TBD for later : downsized V can be used in the Attention and upsized Output matrix in Multi-Head
          

          #W_q(self.query) is num_heads, vecDim, vecDim (look at it as transforming each token into a new linear space)
          # W_k(self.key) is also num_heads, vecDim, vecDim  (look at it as transforming each token into a new linear space)
          # W_v(self.value) is also num_heads, vecDim, vecDim
 
          # Q, K are B H W V. W==S V==I==d_model(inner) same dims as W*

          # Q @ K.T becomes batch_size, num_heads, window_size, window_size ( all pair of token's get an attention score)
             # pytorch uses colom-vectors. math uses row-vectors. 
             # SoftMax is along dim=-1.  SoftMax answers about the query - i.e which token is the highest for this query.
          # V is batch, num_heads, window_size, vecDim

        # Nuance: pytorch matches last two dimensions. 
          # batch_size, num_heads, window_size, window_size @ num_heads, window_size, vecDim works in pytorch, not in maths!

        head_dim = self.keys.shape[-1]
        scores =  (Q @ K.transpose(-2,-1) / (head_dim ** 0.5))
        cur_len = scores.shape[-1]
        if cur_len == self.mask.shape[0]:
            scores.masked_fill_(self.mask, float('-inf'))
        else:
            # inference with context length different from training window — build mask on the fly
            dynamic_mask = torch.triu(torch.ones(cur_len, cur_len, device=scores.device), diagonal=1).bool()
            scores.masked_fill_(dynamic_mask, float('-inf'))
 
        ret_candidate = nn.functional.softmax( scores , dim=-1)  @ V
        return ret_candidate.permute(0,2,1,3).flatten(start_dim=2)

    def dbg_output_health_check(self):
        """Return list of (name, cond, n_inactive, total) for each attention matrix."""
        DEAD = 1e-6
        out  = []
        if args.qkv == 'unfused':
            for name, param in [('Q', self.query), ('K', self.keys), ('V', self.value)]:
                W    = param.detach().cpu().float()
                H, D, hd = W.shape
                S    = torch.linalg.svdvals(W.permute(1, 0, 2).reshape(D, H * hd))
                n    = (S <= DEAD).sum().item()
                cond = (S[0] / S[-1]).item() if S[-1] > 0 else float('inf')
                out.append((name, cond, n, len(S)))
        else:
            W    = self.qkv.detach().cpu().float()
            S    = torch.linalg.svdvals(W)
            n    = (S <= DEAD).sum().item()
            cond = (S[0] / S[-1]).item() if S[-1] > 0 else float('inf')
            out.append(('qkv', cond, n, len(S)))
        return out


