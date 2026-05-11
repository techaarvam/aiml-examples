import torch
import numpy as np
from debug import *
from argParser import *
from torch import nn
from attention import *
import DataInput
import common

class MultiHead (nn.Module):
    def __init__(self):
        super().__init__()

        vecDims = common.vecDims
        usedVocabVecs = []
        for w in common.wordDict.keys():
            if w in common.wordVecs:
                usedVocabVecs.append( common.wordVecs[w] )
            else: usedVocabVecs.append ( np.zeros(vecDims-1) )

        self.register_buffer('wv', torch.tensor(np.array(usedVocabVecs), dtype = torch.float32))
        self.embedding = nn.Embedding( vocabSize, vecDims ) 
        self.embedding.weight.data.copy_(self.wv)

        self.attentionHeads = nn.ModuleList([ Attention() for _ in range (0, args.num_layers) ])
        # Shape Notes:
           # Wo works with concatenated attention heads. reduces to VecDims
           # Wo is the learned re-combination weights, instead of a simple mean.

        self.Wo = nn.ParameterList([nn.Parameter(torch.randn(common.vecDims * args.num_heads, common.vecDims) * (1.0 / (common.vecDims * args.num_heads) ** 0.5)) for _ in range(args.num_layers)])
        if (args.use_custom_norm):
            self.learnedMeanShift = nn.ParameterList([nn.Parameter(torch.zeros(1,1,vecDims)) for _ in range(args.num_layers)])
            self.learnedStdScale  = nn.ParameterList([nn.Parameter(torch.ones(1,1,vecDims))  for _ in range(args.num_layers)])
        else:
            self.norm = nn.ModuleList([nn.LayerNorm(vecDims) for _ in range(args.num_layers)])

        # FFN per token using the same MLP block (same weights)! 
        # Input shape for this FFN: attentionOuputs's tokens + X residual's tokens, so 2*vecDims
        # This MLP can help map attention space to token space
        
        self.mlp = nn.ModuleList( [ nn.Sequential (
                nn.Linear(vecDims * 2, vecDims),
                nn.ReLU(inplace=True)
            )  for _ in range(args.num_layers) ] )
        if (args.output_type == "indices"):
            self.outputLinear = nn.Linear(vecDims, common.vocabSize)
        elif (args.output_type == "vecs"):
            self.outputLinear = nn.Linear(vecDims, vecDims-1)
        
            

    def forward (self, X):
        #Shape Notes:
         # X is batch_size, window_size, vecDim
         # its expanded to include num_heads

        residual = X
        for layer in range (0, args.num_layers):
            # batch_size dim in expand is kept as -1 mindful of the last batch which may not be batch_size.
            attentionOutput = X.unsqueeze(1).expand(-1, args.num_heads, -1, common.vecDims) 

            attentionOutput = self.attentionHeads[layer].forward ( attentionOutput)
            # attentionOutput is batch_size, num_heads, window_size, vecDim
            # lets convert this to batch_size, window_size, num_heads * vecDim, so Wo (output weigts can do a weighted add)
            attentionOutput = attentionOutput.permute(0,2,1,3).flatten(start_dim=2)
            attentionOutput = attentionOutput @ self.Wo[layer]

            # Instead of using nn.LayerNorm, using mean/std operations, 
            # since this is a learning project and the goal is to break
            # the transformer down to simplest operations possible.

            if (args.use_custom_norm):
                mean = attentionOutput.mean(dim=-1, keepdim=True)
                std = attentionOutput.std(dim=-1, keepdim=True)
                attentionOutput = self.learnedStdScale[layer] * (attentionOutput - mean) / (std + 1e-6) + self.learnedMeanShift[layer]
                mean = residual.mean(dim=-1, keepdim=True)
                std = residual.std(dim=-1, keepdim=True)
                residual = self.learnedStdScale[layer] * (residual - mean) / (std + 1e-6) + self.learnedMeanShift[layer]
            else:
                attentionOutput = self.norm[layer](attentionOutput)
                residual = self.norm[layer](residual)


            attentionOutput = self.mlp[layer] ( torch.cat ( (attentionOutput, residual), dim=2) )
            residual = attentionOutput
            X = attentionOutput
            # X = X + attentionOutput Residual is concatenated to the MLP and weigted residual mix
            # happens, so X = X + attentionOutput is not needed

        if (args.output_type == "indices"):
            # Its nice that both the mlp above and the outputLinear below are broadcasting correctly.
            # attentionOuput is batch_size, window_size, vecDims, but outputLinear is vecDims, vocabSize!
            return self.outputLinear(attentionOutput)
        elif (args.output_type == "vecs"):
            return self.outputLinear(attentionOutput) # was [..., :-1] @ self.wv.T # wv is a Tensor from: common.wordVecs

