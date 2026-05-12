
# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
import sys
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

start_epoch = 0
checkpoint = None

if args.model_file:
    checkpoint = torch.load(args.model_file, map_location=common.device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        transformer.load_state_dict(checkpoint['model'])
        transformer = transformer.to(common.dtype)
        dbg_output(f"Loaded model from {args.model_file} (saved after epoch {checkpoint.get('epoch', '?')})")
    else:
        transformer.load_state_dict(checkpoint)
        transformer = transformer.to(common.dtype)
        checkpoint = None  # old format, no optimizer/epoch state
        dbg_output(f"Loaded model from {args.model_file}")

if not args.model_file or args.resume:
    if args.output_type == "indices":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params=transformer.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(params=transformer.parameters(), lr=args.lr)

    if args.resume and checkpoint and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        dbg_output(f"Resuming from epoch {start_epoch + 1}")

    if args.start_epoch is not None:
        start_epoch = args.start_epoch - 1
        dbg_output(f"Starting epoch overridden to {args.start_epoch}")

    if args.lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold=1e-3)
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    else:
        scheduler = None

    interactive       = sys.stdout.isatty()
    current_batch_size = args.batch_size

    for i in range(start_epoch, args.epochs):
        while True:
            try:
                train_loader    = DataLoader(dIn, batch_size=current_batch_size, shuffle=True)
                total_batches   = len(train_loader)
                heartbeat_every = max(1, total_batches // 10)

                dbg_output(f"Epoch {i+1}/{args.epochs} starting... (batch_size={current_batch_size})")
                total_loss = 0.0
                num_batches = 0

                loader_iter = tqdm(train_loader, desc=f"Epoch {i+1}/{args.epochs}", unit="batch") \
                              if interactive else train_loader
                for arg1, arg2, arg3 in loader_iter:
                    if args.embedding_type == "glove-fixed":
                        dInputs        = arg1.to(common.device).to(common.dtype)
                        dLabels        = arg2.to(common.device).to(common.dtype)
                        dTargetIndices = arg3.to(common.device)
                    else:
                        dInputs        = arg1.to(common.device)
                        dLabels        = None
                        dTargetIndices = arg3.to(common.device)

                    optimizer.zero_grad()
                    output = transformer.forward(dInputs)

                    if args.output_type == "indices":
                        B, S, V = output.shape
                        loss = loss_fn(output.reshape(B * S, V), dTargetIndices.reshape(B * S))
                    elif args.output_type == "vecs":
                        loss = loss_fn(output, dLabels[...,:-1])

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                    if not interactive and num_batches % heartbeat_every == 0:
                        dbg_output(f"  [{num_batches}/{total_batches}] loss={total_loss/num_batches:.4f}")

                break  # epoch completed

            except torch.cuda.OutOfMemoryError:
                new_batch = int(current_batch_size * 0.9)
                if new_batch < 16:
                    raise RuntimeError(f"OOM at batch_size={current_batch_size}, cannot reduce further")
                torch.cuda.empty_cache()
                dbg_output(f"OOM at batch_size={current_batch_size} — retrying epoch {i+1} with batch_size={new_batch}")
                current_batch_size = new_batch

        dbg_output(f" Epoch{i+1}: Loss={total_loss/num_batches:.4f}")
        if scheduler:
            if args.lr_schedule == 'plateau':
                scheduler.step(total_loss/num_batches)
            else:
                scheduler.step()
            dbg_output(f" LR={optimizer.param_groups[0]['lr']:.6f}")
        torch.save({'model': transformer.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': i+1}, args.save_model)
        dbg_output(f"Checkpoint saved to {args.save_model}")


infer_ctx = args.infer_window_size if args.infer_window_size is not None else args.window_size

# Run inference only in explicit inference mode (--model_file without --resume)
# or when running interactively. Skip when piped (runner/log mode).
inference_mode = (args.model_file and not args.resume)
if not inference_mode and not sys.stdin.isatty():
    dbg_output("Non-interactive mode — skipping inference prompt. Use cmd_infer.sh to run inference.")
    sys.exit(0)

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
