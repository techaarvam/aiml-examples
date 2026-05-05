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

class DataInput():
    def __init__(self):
        self.wordVecs = api.load("glove-wiki-gigaword-100")

        f=open(args.input, "r")
        text = f.read()
        tokens = word_tokenize (text.lower())

        vectors = []

        self.vecDims = self.wordVecs.vector_size
        debug (f"Dimension of the vector space is: {self.vecDims} ")

        # For a learning curiosity the following gives the same result: 
        #debug (f"Dimension of the vector space is: {wordVecs[word_tokenize('test')].size}")

        for token in tokens:
            if token in self.wordVecs:
                vectors.append(self.wordVecs[token])
            else:
                # Is it really needed to zero-vec out not-found tokens?
                vectors.append(np.zeros(100))

        self.vectors = torch.tensor(np.array(vectors), dtype = torch.float32)

    def __len__(self):
        return len(self.vectors) - args.window_size

    def __getitem__(self, index):
        data =  self.vectors[index: index+args.window_size ]
        label = self.vectors[index+args.window_size]
        return (data,label)

    def getInputSize(self):
        return self.vecDims

    def getWordVecs(self):
        return self.wordVecs

if __name__ == "__main__":
    set_verbosity(DEBUG)
    d = DataInput()


