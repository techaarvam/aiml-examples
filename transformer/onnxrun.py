
# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
# Interactive inference using an ONNX model.
# Usage:
#   python onnxrun.py --model_file transformer.onnx \
#                     --vocab_file vocab.json \
#                     --window_size 64 --output_type indices
# --------------------------------------------------

import numpy as np
import torch
import onnxruntime as ort
from nltk.tokenize import word_tokenize
import DataInput
import common
from argParser import *
from debug import *

set_verbosity(args.verbosity)

if not args.model_file:
    raise ValueError("--model_file is required (path to .onnx file)")

dIn = DataInput.DataInput()

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if common.device == "cuda" else ["CPUExecutionProvider"]
sess = ort.InferenceSession(args.model_file, providers=providers)
dbg_output(f"Loaded ONNX model from {args.model_file}")
dbg_output(f"Running on: {sess.get_providers()[0]}")

infer_ctx = args.infer_window_size if args.infer_window_size is not None else args.window_size

while True:
    dbg_output(f"Enter the starting portion of the text to predict the next words (context={infer_ctx-1} tokens):")
    userInput = input()

    inputTokens = word_tokenize(userInput.lower())
    inputTokens = inputTokens[:infer_ctx-1]
    inputVecs, _ = dIn.tokensToVecsAndIndices(inputTokens)

    pad_len = (infer_ctx-1) - len(inputTokens)
    if pad_len > 0:
        padding = torch.zeros(pad_len, inputVecs.shape[1])
        inputVecs = torch.cat((padding, inputVecs), dim=0)

    generated = list(inputTokens)

    for _ in range(args.output_size):
        infInputs = dIn.embedPositions(inputVecs, seq_len=infer_ctx).unsqueeze(0)
        infInputs_np = infInputs.numpy().astype(np.float32)

        logits = sess.run(["logits"], {"input": infInputs_np})[0]  # [1, seq_len, vocab_size]

        logits_t = torch.tensor(logits[0, -1, :])
        unk_idx = dIn.wordDict.get('<unk>', -1)
        if unk_idx >= 0:
            logits_t[unk_idx] = float('-inf')
        top_logits, top_indices = torch.topk(logits_t, 40)
        probs = torch.softmax(top_logits, dim=-1)
        nextIdx = top_indices[torch.multinomial(probs, 1).item()].item()
        nextWord = dIn.indicesToTokens([nextIdx])[0]
        generated.append(nextWord)

        newVec, _ = dIn.tokensToVecsAndIndices([nextWord])
        inputVecs = torch.cat((inputVecs[1:], newVec), dim=0)

    print(" ".join(generated))
