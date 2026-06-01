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


def _tokenize_file(path, enc, max_tokens, start_epoch, data_stride):
    """Tokenize one file and return (indices_tensor, data_stride). Thread-safe."""
    text   = open(path).read()
    tokens = enc.encode(text)
    indices = torch.tensor(tokens, dtype=torch.long)
    dbg_output(f"Tokenized {len(indices):,} tokens")
    if max_tokens and len(indices) > max_tokens:
        slice_idx = (start_epoch or 0) // 5
        start     = slice_idx * max_tokens
        if start + max_tokens > len(indices):
            start = 0
        indices = indices[start : start + max_tokens]
        dbg_output(f"Capped to {max_tokens:,} tokens, slice {slice_idx} offset {start:,}")
    dbg_output(f"Dataset window stride: {data_stride}")
    return indices


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
                if args.data_stride < 1:
                    raise ValueError("--data_stride must be >= 1")
                self.indices = _tokenize_file(
                    args.input, self.enc, args.max_tokens,
                    args.start_epoch, args.data_stride)
                self.vectors = None
                if args.cache_file:
                    pickle.dump({'indices': self.indices}, open(args.cache_file, 'wb'))
                    dbg_output(f"Saved cache to {args.cache_file}")
            self.data_stride = args.data_stride
        # inference-only mode: enc already initialized above, no indices needed

    def indicesToTokens(self, indices):
        return [self.enc.decode([int(i)]) for i in indices]

    def embedPositions(self, vecs, seq_len=None):
        position_data = torch.arange(0, len(vecs)).unsqueeze(1).float()
        position_data /= (seq_len if seq_len is not None else args.window_size)
        return torch.cat((vecs, position_data), dim=1)

    def __len__(self):
        max_start = len(self.indices) - args.window_size
        if max_start <= 0:
            return 0
        return (max_start + self.data_stride - 1) // self.data_stride

    def __getitem__(self, index):
        start = index * self.data_stride
        input_indices  = self.indices[start : start + args.window_size - 1]
        target_indices = self.indices[start+1 : start + args.window_size]
        return (input_indices, target_indices, target_indices)


if __name__ == "__main__":
    set_verbosity(DEBUG)
    d = DataInput()
