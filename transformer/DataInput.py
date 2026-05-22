# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
import tiktoken
from argParser import *
from debug import *
import numpy as np
import torch
import common
import os
import pickle

_enc_cache = {}

class DataInput():

    def __init__(self):
        self.vecDims = args.vecDims
        common.vecDims   = args.vecDims
        common.innerDims = args.inner_dims if args.inner_dims else args.vecDims
        self.wordVecs = {}
        common.wordVecs = {}

        enc_name = args.tiktoken_encoding
        if enc_name not in _enc_cache:
            _enc_cache[enc_name] = tiktoken.get_encoding(enc_name)
        self.enc = _enc_cache[enc_name]
        self.vocabSize = self.enc.n_vocab
        common.vocabSize = self.vocabSize
        dbg_output(f"tiktoken {enc_name}: vocab size {self.vocabSize:,}")

        if args.input:
            if args.cache_file and os.path.exists(args.cache_file):
                data = pickle.load(open(args.cache_file, 'rb'))
                self.indices = data['indices']
                self.vectors = None
                dbg_output(f"Loaded cache from {args.cache_file} ({len(self.indices):,} tokens)")
            else:
                text = open(args.input).read()
                tokens = self.enc.encode(text)
                self.vectors = None
                self.indices = torch.tensor(tokens, dtype=torch.long)
                dbg_output(f"Tokenized {len(self.indices):,} tokens")
                if args.cache_file:
                    pickle.dump({'indices': self.indices}, open(args.cache_file, 'wb'))
                    dbg_output(f"Saved cache to {args.cache_file}")
            if args.max_tokens and len(self.indices) > args.max_tokens:
                slice_idx = (args.start_epoch or 0) // 5
                start     = slice_idx * args.max_tokens
                if start + args.max_tokens > len(self.indices):
                    start = 0
                self.indices = self.indices[start : start + args.max_tokens]
                dbg_output(f"Capped to {args.max_tokens:,} tokens, slice {slice_idx} offset {start:,}")
        # inference-only mode: enc already initialized above, no indices needed

    def indicesToTokens(self, indices):
        return [self.enc.decode([int(i)]) for i in indices]

    def embedPositions(self, vecs, seq_len=None):
        position_data = torch.arange(0, len(vecs)).unsqueeze(1).float()
        position_data /= (seq_len if seq_len is not None else args.window_size)
        return torch.cat((vecs, position_data), dim=1)

    def __len__(self):
        return len(self.indices) - args.window_size

    def __getitem__(self, index):
        input_indices  = self.indices[index : index + args.window_size - 1]
        target_indices = self.indices[index+1 : index + args.window_size]
        return (input_indices, target_indices, target_indices)


if __name__ == "__main__":
    set_verbosity(DEBUG)
    d = DataInput()
