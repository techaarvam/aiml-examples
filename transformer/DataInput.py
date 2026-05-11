# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
from nltk.tokenize import word_tokenize
import gensim.downloader as api
from argParser import *
from debug import *
import numpy as np
import torch
import common

class DataInput():

    def __init__(self):
        self.wordVecs = api.load("glove-wiki-gigaword-100")
        common.wordVecs = self.wordVecs

        self.vecDims = self.wordVecs.vector_size
        common.vecDims = self.vecDims + 1 # The plus 1 for the position embedding
        dbg_output (f"Dimension of the vector space is: {self.vecDims} ")

        if args.input:
            # training mode: build vocab from text file
            f = open(args.input, "r")
            text = f.read()
            tokens = word_tokenize(text.lower())

            self.vocab = sorted(set(tokens))
            self.wordDict = {w: i for i, w in enumerate(self.vocab)}
            self.vocabSize = len(self.vocab)
            common.vocabSize = self.vocabSize
            common.wordDict = self.wordDict
            dbg_output(f"Vocabulary size: {self.vocabSize}")

            self.vectors, self.indices = self.tokensToVecsAndIndices(tokens)
            self.save_vocab(args.vocab_file)
        elif args.vocab_file:
            # inference mode: load vocab from saved JSON
            self._load_vocab(args.vocab_file)
        else:
            raise ValueError("Either --input (training) or --vocab_file (inference) must be provided")

    def save_vocab(self, path):
        import json
        json.dump(self.wordDict, open(path, "w"))
        dbg_output(f"Vocabulary saved to {path}")

    def _load_vocab(self, path):
        import json
        self.wordDict = json.load(open(path))
        self.vocab = sorted(self.wordDict, key=self.wordDict.get)
        self.vocabSize = len(self.vocab)
        common.vocabSize = self.vocabSize
        common.wordDict = self.wordDict
        dbg_output(f"Vocabulary loaded from {path} ({self.vocabSize} words)")

    # There are tokens, vectors, indices
    # lets have utility functions to handle the conversions 
    def tokensToVecsAndIndices (self, tokens):
        outVecs = []
        indices = []
        for token in tokens:
            if token in self.wordVecs :
                outVecs.append ( self.wordVecs[token] )
            else: outVecs.append ( np.zeros(common.vecDims-1) )
            indices.append(self.wordDict.get(token, 0))

        return  ( torch.tensor(np.array(outVecs), dtype = torch.float32) , torch.tensor ( indices, dtype = torch.long))


    def vecsToTokens (self, vecs):
        # TBD implement if using vector outputs (use_linear_vocab = False)
        dbg_output(f"ERROR: unimplemented")
        pass

    def indicesToTokens (self, indices):
        tokens = []
        for i in indices:
            tokens.append ( self.vocab[int(i)] )
        return tokens

    def embedPositions(self, vecs, seq_len=None):
        # seq_len overrides the normalization denominator for inference with larger context
        position_data = torch.arange(0, len(vecs)).unsqueeze(1).float()
        position_data /= (seq_len if seq_len is not None else args.window_size)

        return torch.cat ((vecs, position_data), dim=1)

    def __len__(self):
        return len(self.vectors) - args.window_size

    def __getitem__(self, index):
        #window_size-1 inputs
        #window_size-1 outputs
        
        data =  self.vectors[index: index+args.window_size ]
        data = self.embedPositions ( data )
       
        target_indices = self.indices[index+1 : index + args.window_size]
        # [:-1] and [1:] suffice, but addingexplicitly 
        return (data[:args.window_size-1] ,data[1:args.window_size], target_indices)

    def getInputSize(self):
        return self.vecDims+1

    def getWordVecs(self):
        return self.wordVecs

if __name__ == "__main__":
    set_verbosity(DEBUG)
    d = DataInput()


