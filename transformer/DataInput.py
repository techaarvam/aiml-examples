# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
from nltk.tokenize import word_tokenize
from collections import Counter
import gensim.downloader as api
from argParser import *
from debug import *
import numpy as np
import torch
import common
import os
import pickle

class DataInput():

    def __init__(self):
        if (args.embedding_type == "glove-fixed"):
            self.wordVecs = api.load("glove-wiki-gigaword-100")
            common.wordVecs = self.wordVecs

            self.vecDims = self.wordVecs.vector_size
            common.vecDims = self.vecDims  # positional encoding added inside model
        else: 
            self.vecDims = args.vecDims
            common.vecDims = args.vecDims
            
            # Keep the vector mapping null if embedding 
            # is learned as part of training. i.e not embedding_type != glove-fixed 
            self.wordVecs = {}
            common.wordVecs = {}

        dbg_output (f"Dimension of the word vector space is: {self.vecDims} ")

        if args.input:
            if args.cache_file and os.path.exists(args.cache_file):
                data = pickle.load(open(args.cache_file, 'rb'))
                self.indices   = data['indices']
                self.wordDict  = data['wordDict']
                self.vocab     = data['vocab']
                self.vocabSize = len(self.vocab)
                common.vocabSize = self.vocabSize
                common.wordDict  = self.wordDict
                self.vectors = None
                dbg_output(f"Loaded cache from {args.cache_file} ({self.vocabSize} words, {len(self.indices):,} tokens)")
                if args.vocab_file:
                    self.save_vocab(args.vocab_file)
            else:
                # training mode: tokenize text, then build or reuse vocab
                f = open(args.input, "r")
                text = f.read()
                tokens = word_tokenize(text.lower())

                if args.vocab_file and os.path.exists(args.vocab_file):
                    self._load_vocab(args.vocab_file)
                    tokens = [t if t in self.wordDict else '<unk>' for t in tokens]
                    dbg_output(f"Re-using vocabulary from {args.vocab_file} ({self.vocabSize} words)")
                else:
                    if args.max_vocab_size > 0:
                        freq = Counter(tokens)
                        top_words = {w for w, _ in freq.most_common(args.max_vocab_size - 1)}
                        tokens = [w if w in top_words else '<unk>' for w in tokens]
                        dbg_output(f"Vocabulary capped at {args.max_vocab_size} (original unique tokens: {len(freq)})")

                    self.vocab = sorted(set(tokens))
                    self.wordDict = {w: i for i, w in enumerate(self.vocab)}
                    self.vocabSize = len(self.vocab)
                    common.vocabSize = self.vocabSize
                    common.wordDict = self.wordDict
                    dbg_output(f"Vocabulary size: {self.vocabSize}")
                    self.save_vocab(args.vocab_file)

                self.vectors = None
                self.indices = torch.tensor(
                    [self.wordDict.get(t, 0) for t in tokens], dtype=torch.long)

                if args.cache_file:
                    pickle.dump({'indices': self.indices, 'wordDict': self.wordDict, 'vocab': self.vocab},
                                open(args.cache_file, 'wb'))
                    dbg_output(f"Saved cache to {args.cache_file}")

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
        # If embedding-type is not glove-fixed self.wordVecs is empty and this 
        # code does not need a change and automatically the outVecs will be empty too. 
        for token in tokens:
            if token in self.wordVecs :
                outVecs.append ( self.wordVecs[token] )
            else: outVecs.append ( np.zeros(common.vecDims) )
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
        return len(self.indices) - args.window_size

    def __getitem__(self, index):
        input_indices  = self.indices[index : index + args.window_size - 1]
        target_indices = self.indices[index+1 : index + args.window_size]

        if args.embedding_type == "glove-fixed":
            tokens_slice = [self.vocab[i.item()] for i in input_indices]
            vecs = [self.wordVecs[t] if t in self.wordVecs else np.zeros(common.vecDims)
                    for t in tokens_slice]
            data = torch.tensor(np.array(vecs), dtype=torch.float32)
            return (data, data, target_indices)
        else:
            return (input_indices, target_indices, target_indices)

    def getInputSize(self):
        return self.vecDims+1

    def getWordVecs(self):
        return self.wordVecs

if __name__ == "__main__":
    set_verbosity(DEBUG)
    d = DataInput()


