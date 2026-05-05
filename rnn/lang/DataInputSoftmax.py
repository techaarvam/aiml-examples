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

class DataInputSoftmax():
    def __init__(self):
        wordVecs = api.load("glove-wiki-gigaword-100")

        f = open(args.input, "r")
        text = f.read()
        tokens = word_tokenize(text.lower())

        self.vecDims = wordVecs.vector_size
        debug(f"Dimension of the vector space is: {self.vecDims}")

        # Build vocab: word -> index
        self.vocab = sorted(set(tokens))
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        dbg_output(f"Vocabulary size: {self.vocab_size}")

        # Build input vectors (GloVe) and label indices
        vectors = []
        indices = []
        for token in tokens:
            if token in wordVecs:
                vectors.append(wordVecs[token])
            else:
                vectors.append(np.zeros(self.vecDims))
            indices.append(self.word_to_idx[token])

        self.vectors = torch.tensor(np.array(vectors), dtype=torch.float32)
        self.indices = torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.vectors) - args.window_size

    def __getitem__(self, index):
        data  = self.vectors[index: index + args.window_size]
        label = self.indices[index + args.window_size]
        return (data, label)

    def getInputSize(self):
        return self.vecDims

    def getVocabSize(self):
        return self.vocab_size

    def getVocab(self):
        return self.vocab

if __name__ == "__main__":
    set_verbosity(DEBUG)
    d = DataInputSoftmax()
