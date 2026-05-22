
# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
import sys
import os
import json
import time
import DataInput
import torch
import numpy as np
from argParser import *
import common
import multihead


from seeder import *
from torch import nn
from torch.utils.data import DataLoader

import random
from debug import *
from tqdm import tqdm

set_seed(args.seed)
set_verbosity (args.verbosity)

common.dtype = {'float32': torch.float32, 'float16': torch.float16,
                'bfloat16': torch.bfloat16, 'float8': torch.float8_e4m3fn}[args.float_type]

def _resolve_input_files():
    if args.input_list:
        if args.input_list.endswith('.json'):
            with open(args.input_list) as f:
                return json.load(f)
        return [p.strip() for p in args.input_list.split(',')]
    return None

_input_files = _resolve_input_files()
_cycling     = _input_files is not None

args.input = _input_files[0] if _cycling else args.input
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

if args.inner_dims:
    _freeze = {'embedding.weight', 'posEmbedding.weight', 'outputLinear.weight', 'outputLinear.bias'}
    for name, param in transformer.named_parameters():
        if name in _freeze:
            param.requires_grad = False
    dbg_output(f"Frozen: embedding, posEmbedding, outputLinear  (inner_dims={args.inner_dims})")

if args.validate:
    import math
    if not args.model_file:
        print("ERROR: --validate requires --model_file")
        sys.exit(1)
    val_loss_fn  = nn.CrossEntropyLoss()
    val_loader   = DataLoader(dIn, batch_size=args.batch_size, shuffle=False)
    total_batches = len(val_loader)
    heartbeat    = max(1, total_batches // 10)
    total_loss   = 0.0
    num_batches  = 0
    transformer.eval()
    print(f"dataset : {args.input}")
    print(f"model   : {args.model_file}")
    print(f"batches : {total_batches}")
    with torch.no_grad():
        for arg1, arg2, arg3 in val_loader:
            if args.embedding_type == "glove-fixed":
                dInputs        = arg1.to(common.device).to(common.dtype)
                dTargetIndices = arg3.to(common.device)
            else:
                dInputs        = arg1.to(common.device)
                dTargetIndices = arg3.to(common.device)
            output = transformer.forward(dInputs)
            B, S, V = output.shape
            loss = val_loss_fn(output.float().reshape(B * S, V), dTargetIndices.reshape(B * S))
            total_loss  += loss.item()
            num_batches += 1
            if num_batches % heartbeat == 0:
                print(f"  [{num_batches}/{total_batches}] loss={total_loss/num_batches:.4f}")
    mean_loss = total_loss / num_batches
    print(f"loss    : {mean_loss:.4f}")
    print(f"ppl     : {math.exp(mean_loss):.2f}")
    sys.exit(0)

if not args.model_file or args.resume:
    if args.output_type == "indices":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    if args.optimizer == "adam":
        # Autograd and gradscale are not used in this code. (AMP is not used)
        # 1e-4 to help numerical stability in bfloat16
        optimizer = torch.optim.Adam(params=transformer.parameters(), lr=args.lr, eps=1e-4)
    else:
        optimizer = torch.optim.SGD(params=transformer.parameters(), lr=args.lr)

    if args.resume and checkpoint and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for pg in optimizer.param_groups:   # add these two lines
            pg['lr'] = args.lr
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

    interactive    = sys.stdout.isatty()
    progress_file  = os.path.join(os.path.dirname(args.save_model), "progress.txt") \
                     if args.save_model else None
    train_loader    = DataLoader(dIn, batch_size=args.batch_size, shuffle=True)
    total_batches   = len(train_loader)
    heartbeat_every = max(1, total_batches // 10)
    _current_file = args.input if _cycling else None

    for i in range(start_epoch, args.epochs):
        if _cycling:
            epoch_file = _input_files[i % len(_input_files)]
            if epoch_file != _current_file:
                _current_file   = epoch_file
                args.input      = epoch_file
                args.cache_file  = None
                args.start_epoch = i
                dbg_output(f"Input [{i % len(_input_files) + 1}/{len(_input_files)}]: {epoch_file}")
                dIn             = DataInput.DataInput()
                train_loader    = DataLoader(dIn, batch_size=args.batch_size, shuffle=True)
                total_batches   = len(train_loader)
                heartbeat_every = max(1, total_batches // 10)
        dbg_output(f"Epoch {i+1}/{args.epochs} starting...")
        total_loss  = 0.0
        num_batches = 0
        epoch_start = time.time()

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
            assert not output.isnan().any(), f"NaN in forward pass: max={output.abs().max()}"
            if args.output_type == "indices":
                B, S, V = output.shape
                loss = loss_fn(output.reshape(B * S, V), dTargetIndices.reshape(B * S))
            elif args.output_type == "vecs":
                loss = loss_fn(output, dLabels[...,:-1])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

            if not interactive and num_batches % heartbeat_every == 0:
                dbg_output(f"  [{num_batches}/{total_batches}] loss={total_loss/num_batches:.4f}")

            if progress_file:
                elapsed   = time.time() - epoch_start
                rate      = num_batches / elapsed
                remaining = (total_batches - num_batches) / rate
                lr_cur    = optimizer.param_groups[0]['lr']
                with open(progress_file, 'w') as pf:
                    pf.write(
                        f"Epoch    : {i+1} / {args.epochs}\n"
                        f"Batch    : {num_batches} / {total_batches}  ({100*num_batches/total_batches:.1f}%)\n"
                        f"Loss     : {total_loss/num_batches:.4f}\n"
                        f"LR       : {lr_cur:.6f}\n"
                        f"Rate     : {rate:.1f} batches/sec\n"
                        f"Elapsed  : {elapsed/60:.1f} min\n"
                        f"ETA epoch: {remaining/60:.1f} min\n"
                        f"Updated  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )

        dbg_output(f" Epoch{i+1}: Loss={total_loss/num_batches:.4f}")
        if scheduler:
            if args.lr_schedule == 'plateau':
                scheduler.step(total_loss/num_batches)
            else:
                scheduler.step()
            dbg_output(f" LR={optimizer.param_groups[0]['lr']:.6f}")
        torch.save({'model': transformer.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': i+1}, args.save_model)
        dbg_output(f"Checkpoint saved to {args.save_model}")


def sample_token(logits):
    logits = logits.float()
    if args.temperature != 1.0:
        logits = logits / args.temperature
    if args.sampler == 'top_k':
        k = min(args.top_k, logits.size(-1))
        top_logits, top_indices = torch.topk(logits, k)
        probs = torch.softmax(top_logits, dim=-1)
        return top_indices[torch.multinomial(probs, 1).item()].item()
    elif args.sampler == 'top_p':
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = (cumsum - sorted_probs) < args.top_p
        keep[0] = True  # always keep top token
        filtered = sorted_probs * keep
        filtered = filtered / filtered.sum()
        return sorted_indices[torch.multinomial(filtered, 1).item()].item()
    else:  # min_p
        probs = torch.softmax(logits, dim=-1)
        threshold = args.min_p * probs.max().item()
        mask = probs >= threshold
        filtered = probs * mask
        filtered = filtered / filtered.sum()
        return torch.multinomial(filtered, 1).item()

infer_ctx = args.infer_window_size if args.infer_window_size is not None else args.window_size

# Run inference only in explicit inference mode (--model_file without --resume)
# or when running interactively. Skip when piped (runner/log mode).
inference_mode = (args.model_file and not args.resume)
if not inference_mode and not sys.stdin.isatty():
    dbg_output("Non-interactive mode — skipping inference prompt. Use cmd_infer.sh to run inference.")
    sys.exit(0)

while True:
    dbg_output(f"Enter the starting portion of the text to predict the next words (context={infer_ctx-1} tokens):")
    userInput = input()

    inputIds = dIn.enc.encode(userInput)
    inputIds = inputIds[-(infer_ctx - 1):]
    pad_len = (infer_ctx - 1) - len(inputIds)
    inputIndices = torch.tensor(inputIds, dtype=torch.long)
    if pad_len > 0:
        padding = torch.zeros(pad_len, dtype=torch.long)
        inputIndices = torch.cat((padding, inputIndices), dim=0)

    generated_ids = list(inputIds)
    with torch.no_grad():
        for _ in range(args.output_size):
            infInputs = inputIndices.unsqueeze(0).to(common.device)
            infOutputs = transformer.forward(infInputs)
            logits = infOutputs[0, -1, :]
            nextIdx = sample_token(logits)
            generated_ids.append(nextIdx)
            newIdx = torch.tensor([nextIdx], dtype=torch.long)
            inputIndices = torch.cat((inputIndices[1:], newIdx))

    print(dIn.enc.decode(generated_ids))
