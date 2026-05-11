
# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
import DataInput
import torch
import numpy as np
from argParser import *
import common
import multihead

from seeder import *
from torch import nn
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
import random
from debug import *
from tqdm import tqdm

set_seed(args.seed)
set_verbosity (args.verbosity)

common.dtype = {'float32': torch.float32, 'float16': torch.float16,
                'bfloat16': torch.bfloat16, 'float8': torch.float8_e4m3fn}[args.float_type]

dIn = DataInput.DataInput()

transformer = multihead.MultiHead ().to(common.device).to(common.dtype)
total_params = sum(p.numel() for p in transformer.parameters())
trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
dbg_output(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

if args.model_file:
    transformer.load_state_dict(torch.load(args.model_file, map_location=common.device))
    transformer = transformer.to(common.dtype)
    dbg_output(f"Loaded Model from {args.model_file}")
else:
    # Lets train!
    #   choice of outputs:
    #      index into the vocabulary (targetIndices) 
    #      vector itself (only projected down from 101 -> 100 dims) 
    # 

    if args.output_type == "indices":
        loss_fn = nn.CrossEntropyLoss()
    else: 
        loss_fn = nn.MSELoss()

    if (args.optimizer == "adam"):
        optimizer = torch.optim.Adam(params = transformer.parameters(), lr = args.lr)
    else:
        optimizer = torch.optim.SGD(params = transformer.parameters(), lr = args.lr)

    if args.lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold=1e-3)
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    else:
        scheduler = None

    train_loader = DataLoader(dIn, batch_size = args.batch_size, shuffle=True)

    for i in range(0, args.epochs):
        dbg_output(f"Epoch {i+1}/{args.epochs} starting...")
        total_loss = 0.0
        num_batches = 0

        
        for arg1, arg2, arg3 in tqdm(train_loader, desc=f"Epoch {i+1}/{args.epochs}", unit="batch"):
            if args.embedding_type == "glove-fixed":
                # arg1=inputs (float vecs), arg2=labels (float vecs), arg3=targetIndices (long)
                dInputs       = arg1.to(common.device).to(common.dtype)
                dLabels       = arg2.to(common.device).to(common.dtype)
                dTargetIndices = arg3.to(common.device)
            else:
                # arg1=inputIndices (long), arg2/arg3=targetIndices (long)
                dInputs       = arg1.to(common.device)   # long indices, no float cast
                dLabels       = None
                dTargetIndices = arg3.to(common.device)

            optimizer.zero_grad()
            output = transformer.forward(dInputs)

            if (args.output_type == "indices"):
                # CrossEntropy expect maths style row-vector (for the last 2 dims), so permuting labels 1,2
                loss = loss_fn(output.permute(0,2,1), dTargetIndices)
            elif (args.output_type == "vecs"):
                loss = loss_fn (output, dLabels[...,:-1])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches = num_batches + 1
 
        dbg_output (f" Epoch{i+1}: Loss={total_loss/num_batches:.4f}")
        if scheduler:
            if args.lr_schedule == 'plateau':
                scheduler.step(total_loss/num_batches)
            else:
                scheduler.step()
            dbg_output(f" LR={optimizer.param_groups[0]['lr']:.6f}")
        torch.save(transformer.state_dict(), args.save_model)
        dbg_output (f"Checkpoint saved to {args.save_model}")

    if (args.save_model):
        torch.save(transformer.state_dict(), args.save_model)
        dbg_output (f"Model saved to {args.save_model}")


infer_ctx = args.infer_window_size if args.infer_window_size is not None else args.window_size

while True:
    dbg_output (f"Enter the starting portion of the text to predict the next words (context={infer_ctx-1} tokens):")
    userInput = input()

    inputTokens = word_tokenize (userInput.lower())

    inputTokens = inputTokens[:infer_ctx-1]
    inputVecs, inputIndices = dIn.tokensToVecsAndIndices(inputTokens)

    pad_len = (infer_ctx-1) - len(inputTokens)
    if args.embedding_type == "glove-fixed":
        if pad_len > 0:
            padding = torch.zeros(pad_len, inputVecs.shape[1])
            inputVecs = torch.cat((padding, inputVecs), dim=0)
    else:
        if pad_len > 0:
            padding = torch.zeros(pad_len, dtype=torch.long)
            inputIndices = torch.cat((padding, inputIndices), dim=0)

    generated = list(inputTokens)
    with torch.no_grad():
        for _ in range(args.output_size):
            if args.embedding_type == "glove-fixed":
                infInputs = inputVecs.unsqueeze(0).to(common.device).to(common.dtype)
            else:
                infInputs = inputIndices.unsqueeze(0).to(common.device)

            infOutputs = transformer.forward(infInputs)

            nextIdx = infOutputs[0, -1, :].argmax().item()
            nextWord = dIn.indicesToTokens([nextIdx])[0]
            generated.append(nextWord)

            if args.embedding_type == "glove-fixed":
                newVec, _ = dIn.tokensToVecsAndIndices([nextWord])
                inputVecs = torch.cat((inputVecs[1:], newVec), dim=0)
            else:
                newIdx = torch.tensor([nextIdx], dtype=torch.long)
                inputIndices = torch.cat((inputIndices[1:], newIdx))

    print(" ".join(generated))
