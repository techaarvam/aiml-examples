import torch
import numpy as np
from debug import *
from argParser import *
from torch import nn
from attention import *
import DataInput
import common
from torch.utils.checkpoint import checkpoint as ckpt

class MultiHead (nn.Module):
    def __init__(self):
        super().__init__()

        vecDims   = common.vecDims
        innerDims = common.innerDims
        isGlove   = args.embedding_type == "glove-fixed"

        if isGlove:
            usedVocabVecs = []
            for w in common.wordDict.keys():
                if w in common.wordVecs:
                    usedVocabVecs.append( common.wordVecs[w] )
                else: usedVocabVecs.append( np.zeros(vecDims) )
            self.register_buffer('wv', torch.tensor(np.array(usedVocabVecs), dtype=torch.float32))
        else:
            self.embedding = nn.Embedding(common.vocabSize, vecDims)

        # positional embedding used by both modes
        self.posEmbedding = nn.Embedding(args.window_size, vecDims)

        # upscale/downscale adapters: bridge embedding dim (vecDims) ↔ transformer dim (innerDims)
        if innerDims != vecDims:
            self.upscale   = nn.Linear(vecDims, innerDims, bias=False)
            self.downscale = nn.Linear(innerDims, vecDims, bias=False)

        self.attentionHeads = nn.ModuleList([ Attention() for _ in range (0, args.num_layers) ])
        # Shape Notes:
           # Wo works with concatenated attention heads. reduces to VecDims
           # Wo is the learned re-combination weights, instead of a simple mean.

        self.Wo = nn.ParameterList([nn.Parameter(torch.randn(innerDims, innerDims) * (1.0 / innerDims ** 0.5)) for _ in range(args.num_layers)])
        if (args.use_custom_norm):
            self.learnedMeanShift1 = nn.ParameterList([nn.Parameter(torch.zeros(1,1,innerDims)) for _ in range(args.num_layers)])
            self.learnedStdScale1  = nn.ParameterList([nn.Parameter(torch.ones(1,1,innerDims))  for _ in range(args.num_layers)])
            self.learnedMeanShift2 = nn.ParameterList([nn.Parameter(torch.zeros(1,1,innerDims)) for _ in range(args.num_layers)])
            self.learnedStdScale2  = nn.ParameterList([nn.Parameter(torch.ones(1,1,innerDims))  for _ in range(args.num_layers)])
        else:
            self.norm1 = nn.ModuleList([nn.LayerNorm(innerDims) for _ in range(args.num_layers)])
            self.norm2 = nn.ModuleList([nn.LayerNorm(innerDims) for _ in range(args.num_layers)])

        # Standard FFN: d -> 4d -> d with GELU
        self.mlp = nn.ModuleList([nn.Sequential(
                nn.Linear(innerDims, innerDims * 4),
                nn.GELU(),
                nn.Linear(innerDims * 4, innerDims)
            ) for _ in range(args.num_layers)])
        if (args.output_type == "indices"):
            self.outputLinear = nn.Linear(vecDims, common.vocabSize)
        elif (args.output_type == "vecs"):
            self.outputLinear = nn.Linear(vecDims, vecDims-1)
        
    def _layer_forward(self, X, layer):
        normedX = self.normalize(X, self.norm1[layer],
            (self.learnedMeanShift1[layer], self.learnedStdScale1[layer]) if args.use_custom_norm else None)
        attentionOutput = normedX.unsqueeze(1).expand(-1, args.num_heads, -1, common.innerDims)
        attentionOutput = self.attentionHeads[layer].forward(attentionOutput)
        attentionOutput = attentionOutput.permute(0, 2, 1, 3).flatten(start_dim=2)
        attentionOutput = attentionOutput @ self.Wo[layer]
        X = X + attentionOutput
        normedX = self.normalize(X, self.norm2[layer],
            (self.learnedMeanShift2[layer], self.learnedStdScale2[layer]) if args.use_custom_norm else None)
        X = X + self.mlp[layer](normedX)
        return X

    def normalize (self, X, norm, learnedParams=None):
        # Instead of using nn.LayerNorm, using mean/std operations,
        # since this is a learning project and the goal is to break
        # the transformer down to simplest operations possible.

        if (args.use_custom_norm):
            learnedMeanShift, learnedStdScale = learnedParams
            mean = X.mean(dim=-1, keepdim=True)
            std = X.std(dim=-1, keepdim=True)
            normedX = learnedStdScale * (X - mean) / (std + 1e-6) + learnedMeanShift
        else:
            normedX = norm(X)
        return normedX


    def forward (self, X):
        #Shape Notes:
         # X is batch_size, window_size, vecDim
         # its expanded to include num_heads

        positions = torch.arange(X.shape[1], device=X.device)
        if args.embedding_type == "glove-fixed":
            X = X + self.posEmbedding(positions)          # X is float GloVe vecs
        else:
            X = self.embedding(X) + self.posEmbedding(positions)  # X is long indices
        if hasattr(self, 'upscale'):
            X = self.upscale(X)
        for layer in range (0, args.num_layers):
            if args.grad_checkpoint and self.training:
                fn = lambda X, l=layer: self._layer_forward(X, l)
                X = ckpt(fn, X, use_reentrant=False)
            else:
                X = self._layer_forward(X, layer)
        if hasattr(self, 'downscale'):
            X = self.downscale(X)
        if (args.output_type == "indices"):
            return self.outputLinear(X)
        elif (args.output_type == "vecs"):
            return self.outputLinear(X)

