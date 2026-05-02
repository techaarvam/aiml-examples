# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
from debug import *
from common import *
import torch

from torch import nn

# Lets implement conv2D grounds up to understand under the hood deeper.
# N, C, H, W is the standard conv2D, lets use the same 
# The simplicity of implementing a model is quite impressive, thanks to requires_grad=True
class MyConv2D(nn.Module):
    # h_out, w_out are assumed == h_in, w_in 
    # c_out == c_in == 1 
    # 

    def __init__(self, H, W, kernel_size):
        super().__init__()
        self.c_in = 1
        self.H = H
        self.W = W
        self.c_out = 1
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(self.c_out, kernel_size, kernel_size))
        self.bias = nn.Parameter (torch.randn( self.c_out))

    def forward(self, x):

        # shapes: bias is c_out
        #         weight is c_out, kernel_size
        #         x is N,c_in, H, W
               # thinking: kernel_size made of a,b, for every x_ij   , a slice of
               # x_ij i till i+a and j till j+b to be hadamard product, flattend and summed
               # taking a,b as kernel_size.
               # c_in and c_out are 1 channel for now. 
        weight_expanded = self.weight.reshape(
                               self.c_out,
                               self.kernel_size,
                               self.kernel_size,
                               1, 1, 1, 1
                              )

        debug(f"weight_expanded shape is : {weight_expanded.shape}")

        # for this approach , lets pad x with zero's on the edges to allow cross-corr to be valid throughout
        # x 

        # left, right, top, bottom padding are the tuple arguments
        x_padded = torch.nn.functional.pad(x, (0, self.kernel_size-1, 0, self.kernel_size-1))
        debug ("shape of x_padded is ", x_padded.shape)

        x_padded = x_padded.unfold(2, self.kernel_size, 1).unfold(3,self.kernel_size,1)
        debug ("shape of x_padded is ", x_padded.shape)
        # x_padded shape after the unfold is N c_in, H, W,k,k
        x_padded = x_padded.unsqueeze(0) # add the c_out
        debug ("shape of x_padded is ", x_padded.shape)
        x_padded = x_padded.permute (0,5,6,1,2,3,4)
        debug ("shape of x_padded is ", x_padded.shape)
        # weight_expanded shape is c_out, k, k, 1, 1, 1, 1
        # Trailing dimensions are matching. This broadcast hadamard product will work?
        # check:

        W_X = torch.sum(weight_expanded * x_padded, dim=(1,2,4))
        W_X = W_X.permute(1,0,2,3) + self.bias.reshape(self.c_out, 1,1,1)

        debug (f"haddamard broadcast magic: W * X: {W_X}, shape: {W_X.shape}")
        return W_X
       

class CNN(nn.Module): 
    # Could have used nn.Sequential too.
    
    def __init__(self, hidden_size):
        super().__init__()
                              #  H, W, k
        self.layer_1 = MyConv2D (28, 28, 2)
        self.layer_2 = nn.ReLU ()
        self.layer_3 = nn.MaxPool2d(2)
        self.layer_4 = nn.Flatten()
        self.layer_5 = nn.Linear (14*14, hidden_size)
        self.layer_6 = nn.ReLU ()
        self.layer_7 = nn.Linear (hidden_size, 2)
        self.layer_8 = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.layer_8 ( 
        self.layer_7 ( 
        self.layer_6 ( 
        self.layer_5 ( 
        self.layer_4 ( 
        self.layer_3 ( 
        self.layer_2 ( 
        self.layer_1 ( x ) )))))))


if __name__ == "__main__":  
    set_verbosity(DEBUG)
                # c_in, H, W, k
    m = MyConv2D (28,28,2)
    x = torch.randn(2,4,4).unsqueeze(1)
    m.forward(x)

