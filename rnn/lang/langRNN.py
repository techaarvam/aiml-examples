# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
import numpy as np
import torch
from debug import *
from argParser import *
from torch import nn

import DataInput

# The top level model (output)
class langRNN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.rg = ResetGate(dim, hidden_dim)
        self.ug = UpdateGate(dim, hidden_dim)
        self.candidateH = CandidateHiddenState(dim, hidden_dim)
        self.dim = dim
        self.hidden_dim = hidden_dim

        # h_prev will not be a parameter, precise state-storage,
        # must not be updated by backprop
        self.register_buffer('h', torch.zeros(args.batch_size, hidden_dim), persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, full_x):
        # full_x will have args.window_size items. 

        # inference batch_size is different, so this hack to not use zeros_like(self.h)
        self.h = torch.zeros(full_x.shape[0], self.hidden_dim, device=full_x.device)
        # zeroout the internal state h for each sequence. 
        # h shape is correctly batch_size, dim: so each item in the batch has its own zeroed out state to work with

        for i in range(0, args.window_size):
            
            x = full_x[:,i,:]
            combined_h_x =  (self.h, x)
            rg_out = self.rg(combined_h_x)
            z_out = self.ug(combined_h_x)

            combined_r_h_x = (rg_out , self.h, x)

            ch_out = self.candidateH (combined_r_h_x)
            self.h = ( 1 - z_out ) * self.h + z_out * ch_out

            # Continue to keep h as non-participant in gradiant graphs requires_grad = False 
            # TBD: really needed?
            self.h = self.h.detach() 
 

            #interim outputs are ignored, the final prediction after the sequence is the net output.
            output = self.mlp(self.h)

        return output
        

class Gates(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # Wr is not a descriptive enough variable name.
        # using since this is directly coming from the maths equation of the reset gate.
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.W_x = nn.Parameter( torch.randn(dim, hidden_dim) )
        self.W_r = nn.Parameter( torch.randn(hidden_dim, hidden_dim) )
        self.bias_r = nn.Parameter( torch.zeros(hidden_dim) ) 

    
    def forward(self, combined_h_x):
        h = combined_h_x[0]
        x = combined_h_x[1]
        resetGateOutput = torch.sigmoid ( x @ self.W_x + h @ self.W_r + self.bias_r) 
        return resetGateOutput

class ResetGate(Gates):
    pass

class UpdateGate(Gates):
    pass


class CandidateHiddenState(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_xh = nn.Parameter(torch.randn(dim, hidden_dim))
        self.bias_h = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, combined_r_h_x):
        r = combined_r_h_x[0]
        h = combined_r_h_x[1]
        x = combined_r_h_x[2]
        h = r * h
        ch_out = torch.tanh ( x @ self.W_xh + h @ self.W_hh  + self.bias_h )
        return ch_out


if __name__ == "__main__":
    pass
    # TBD: code smoke tests
    #l = langRNN(dim) 
