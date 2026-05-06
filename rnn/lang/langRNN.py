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
    def __init__(self, dim, hidden_dim, vocab_size=None):
        super().__init__()

        # The output from the first layer is hidden_dim wide, so this
        # if layer == 0 dim, hidden_dim otherwise hidden_dim, hidden_dim

        self.rg = nn.ModuleList([ResetGate(dim if layer==0 else hidden_dim, hidden_dim)
                                    for layer in range(args.num_layers)])
        self.ug = nn.ModuleList([UpdateGate(dim if layer == 0 else hidden_dim, hidden_dim) 
                                    for layer in range(args.num_layers)])
        self.candidateH = nn.ModuleList( [ CandidateHiddenState(dim if layer==0 else hidden_dim, hidden_dim)
                                    for layer in range(args.num_layers)])
        self.dim = dim
        self.hidden_dim = hidden_dim


        # softmax mode: output over vocab; glove mode: output into embedding space
        output_size = vocab_size if vocab_size is not None else dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, full_x):
        # full_x will have args.window_size items. 

        # inference batch_size is different, so this hack to not use zeros_like(self.h)
        h = []
        for layer in range(args.num_layers):
            h.append(torch.zeros(full_x.shape[0], self.hidden_dim, device = full_x.device ))

        # zeroout the internal state h for each sequence. 
        # h shape is correctly batch_size, dim: so each item in the batch has its own zeroed out state to work with

        for i in range(0, args.window_size):
            
            x = full_x[:,i,:]
            for layer in range(0, args.num_layers):

                combined_h_x =  (h[layer], x)
                rg_out = self.rg[layer](combined_h_x)
                z_out = self.ug[layer](combined_h_x)

                combined_r_h_x = (rg_out , h[layer], x)

                ch_out = self.candidateH[layer] (combined_r_h_x)

                h[layer] = ( 1 - z_out ) * h[layer] + z_out * ch_out
                x = h[layer]

 


        return self.mlp(h[args.num_layers-1])
        

class Gates(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # Wr is not a descriptive enough variable name.
        # using since this is directly coming from the maths equation of the reset gate.
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Attempt to limit gradient explosion, start with small weights
        self.W_x = nn.Parameter( torch.randn(dim, hidden_dim)*0.01 )
        self.W_r = nn.Parameter( torch.randn(hidden_dim, hidden_dim)*0.01 )
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

        # Attempt to limit gradient explosion, start with small weights
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) *0.01)
        self.W_xh = nn.Parameter(torch.randn(dim, hidden_dim) *0.01)
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
