
# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
import sys
import os
import json
import time
import concurrent.futures
import DataInput
import torch
import numpy as np
from argParser import *
import common
import multihead
from bitsandbytes.optim import Adam8bit


from torch.profiler import profile, ProfilerActivity, schedule

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

torch.set_float32_matmul_precision('high')


def _resolve_input_files():
    if args.input_list:
        if args.input_list.endswith('.json'):
            with open(args.input_list) as f:
                return json.load(f)
        return [p.strip() for p in args.input_list.split(',')]
    return None

_input_files = _resolve_input_files()
_cycling     = _input_files is not None

if _cycling:
    _init_epoch = (args.start_epoch - 1) if args.start_epoch is not None else 0
    args.input = _input_files[_init_epoch % len(_input_files)]
dIn = DataInput.DataInput()

transformer = multihead.MultiHead ().to(common.device).to(common.dtype)
total_params = sum(p.numel() for p in transformer.parameters())
trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
dbg_output(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

start_epoch = 0
checkpoint = None

def _remap_state_dict(sd):
    import re
    out = {}
    for k, v in sd.items():
        k = k.replace('_orig_mod.', '')
        # Sequential MLP keys (mlp.L.0/2.*) → _MLP keys (mlp.L.fc1/fc2.*)
        k = re.sub(r'^(mlp\.\d+)\.0\.(weight|bias)$', r'\1.fc1.\2', k)
        k = re.sub(r'^(mlp\.\d+)\.2\.(weight|bias)$', r'\1.fc2.\2', k)
        out[k] = v
    return out

if args.model_file:
    checkpoint = torch.load(args.model_file, map_location=common.device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = _remap_state_dict(checkpoint['model'])
        transformer.load_state_dict(state_dict)
        transformer = transformer.to(common.dtype)
        dbg_output(f"Loaded model from {args.model_file} (saved after epoch {checkpoint.get('epoch', '?')})")
    else:
        state_dict = _remap_state_dict(checkpoint)
        transformer.load_state_dict(state_dict)
        transformer = transformer.to(common.dtype)
        checkpoint = None  # old format, no optimizer/epoch state
        dbg_output(f"Loaded model from {args.model_file}")

if args.inner_dims:
    _freeze = {'embedding.weight', 'posEmbedding.weight', 'outputLinear.weight', 'outputLinear.bias'}
    for name, param in transformer.named_parameters():
        if name in _freeze:
            param.requires_grad = False
    dbg_output(f"Frozen: embedding, posEmbedding, outputLinear  (inner_dims={args.inner_dims})")

if not args.validate:
    torch._dynamo.config.capture_scalar_outputs = True
    transformer = torch.compile(transformer, mode="default")
    dbg_output("torch.compile: default mode")

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
    if args.output_type != "indices":
        loss_fn = nn.MSELoss()

    if args.optimizer == "adam":
        # Autograd and gradscale are not used in this code. (AMP is not used)
        # 1e-4 to help numerical stability in bfloat16
        optimizer = torch.optim.Adam(params=transformer.parameters(), lr=args.lr, eps=1e-4)
    elif args.optimizer == "adam8":
        optimizer = Adam8bit(transformer.parameters(), lr = args.lr)
    else:
        optimizer = torch.optim.SGD(params=transformer.parameters(), lr=args.lr, eps=1e-4)

    if args.resume and checkpoint and 'optimizer' in checkpoint and not args.reset_optimizer_every_epoch:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for pg in optimizer.param_groups:   # add these two lines
            pg['lr'] = args.lr
        start_epoch = checkpoint.get('epoch', 0)
        dbg_output(f"Resuming from epoch {start_epoch + 1}")

    if args.start_epoch is not None:
        start_epoch = args.start_epoch - 1
        dbg_output(f"Starting epoch overridden to {args.start_epoch}")

    if _cycling:
        _correct_file = _input_files[start_epoch % len(_input_files)]
        if _correct_file != args.input:
            args.input      = _correct_file
            args.cache_file = None
            dIn             = DataInput.DataInput()
            dbg_output(f"Pre-loaded input [{start_epoch % len(_input_files) + 1}/{len(_input_files)}]: {_correct_file}")

    if args.lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, threshold=1e-3)
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    else:
        scheduler = None

    interactive    = sys.stdout.isatty()
    progress_file  = os.path.join(os.path.dirname(args.save_model), "progress.txt") \
                     if args.save_model else None
    _dl_kwargs = dict(batch_size=args.batch_size, shuffle=True,
                      num_workers=args.dataloader_workers,
                      prefetch_factor=args.prefetch_factor,
                      persistent_workers=args.dataloader_workers > 0)
    train_loader    = DataLoader(dIn, **_dl_kwargs)
    total_batches   = len(train_loader)
    heartbeat_every = max(1, total_batches // 10)
    _current_file = args.input if _cycling else None
    _prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _prefetch_future: concurrent.futures.Future | None = None


    def reset_adam_v(opt):
        """Zero exp_avg_sq (v_t) for all params while keeping exp_avg (m_t)."""
        for group in opt.param_groups:
            for p in group['params']:
                state = opt.state[p]
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].zero_()

    def dump_profile(prof):
        prof.export_chrome_trace("trace.json")

    with profile ( activities=[ ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   schedule = schedule(wait=2, warmup=1, active=3, repeat=1),
                   record_shapes = True,
                   on_trace_ready = dump_profile,
                   profile_memory = True,
                   with_stack = True ) as prof:


        for i in range(start_epoch, args.epochs):
            if _cycling:
                epoch_file = _input_files[i % len(_input_files)]
                if epoch_file != _current_file:
                    _current_file = epoch_file
                    dbg_output(f"Input [{i % len(_input_files) + 1}/{len(_input_files)}]: {epoch_file}")
                    if _prefetch_future is not None and _prefetch_future.done():
                        # use pre-tokenized data from background thread
                        indices = _prefetch_future.result()
                        _prefetch_future = None
                        dIn.indices    = indices
                        dIn.data_stride = args.data_stride
                    else:
                        # fallback: tokenize synchronously (first epoch or future not ready)
                        if _prefetch_future is not None:
                            indices = _prefetch_future.result()  # wait for it
                            _prefetch_future = None
                            dIn.indices    = indices
                            dIn.data_stride = args.data_stride
                        else:
                            args.input      = epoch_file
                            args.cache_file = None
                            args.start_epoch = i
                            dIn = DataInput.DataInput()
                    train_loader    = DataLoader(dIn, **_dl_kwargs)
                    total_batches   = len(train_loader)
                    heartbeat_every = max(1, total_batches // 10)

            # kick off pre-tokenization for next epoch while this one trains
            if _cycling:
                next_file = _input_files[(i + 1) % len(_input_files)]
                if next_file != _current_file and _prefetch_future is None:
                    dbg_output(f"Pre-fetching next input: {next_file}")
                    _prefetch_future = _prefetch_executor.submit(
                        DataInput._tokenize_file,
                        next_file, dIn.enc, args.max_tokens, i + 1, args.data_stride)
            _ts = time.strftime('%Y-%m-%d %H:%M:%S')
            dbg_output(f"Epoch {i+1}/{args.epochs} starting...  [{_ts}]")
            if i == start_epoch and torch.cuda.is_available():
                _vram_alloc = torch.cuda.memory_allocated() / 1024**3
                _vram_res   = torch.cuda.memory_reserved()  / 1024**3
                dbg_output(f"  VRAM: {_vram_alloc:.2f} GB allocated / {_vram_res:.2f} GB reserved")
                try:
                    import subprocess as _sp
                    _smi = _sp.check_output(
                        ['nvidia-smi', '--query-gpu=power.draw,power.limit',
                         '--format=csv,noheader,nounits'],
                        text=True).strip()
                    _draw, _limit = [x.strip() for x in _smi.split(',')]
                    dbg_output(f"  Power: {_draw} W draw / {_limit} W limit")
                except Exception:
                    pass
            if args.reset_adam_v_every_epoch:
                reset_adam_v(optimizer)
                dbg_output(f"  Adam v_t reset: exp_avg_sq zeroed, exp_avg kept, lr={args.lr}")

            if args.reset_optimizer_every_epoch and i > start_epoch:
                if args.optimizer == "adam":
                    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=args.lr, eps=1e-4)
                elif args.optimizer == "adam8":
                    optimizer = Adam8bit(transformer.parameters(), lr=args.lr)
                else:
                    optimizer = torch.optim.SGD(params=transformer.parameters(), lr=args.lr, eps=1e-4)
                if args.lr_schedule == 'plateau':
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, threshold=1e-3)
                elif args.lr_schedule == 'cosine':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
                else:
                    scheduler = None
                dbg_output(f"  Optimizer reset: fresh state, lr={args.lr}")
            total_loss  = 0.0
            num_batches = 0
            epoch_start = time.time()
    
            loader_iter = tqdm(train_loader, desc=f"Epoch {i+1}/{args.epochs}", unit="batch") \
                          if interactive else train_loader

            if (args.lr_warmup_target > 0):
                increase_lr_by = args.lr_warmup_target - args.lr
                increase_per_batch =  increase_lr_by / (total_batches / 10)

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
            
                if args.output_type == "indices":
                    loss = transformer.forward(dInputs, dTargetIndices)
                    if isinstance(loss, torch.Tensor):
                        loss.backward()
                        loss = loss.item()
                elif args.output_type == "vecs":
                    output = transformer.forward(dInputs)
                    assert not output.isnan().any(), f"NaN in forward pass: max={output.abs().max()}"
                    loss_t = loss_fn(output, dLabels[...,:-1])
                    loss_t.backward()
                    loss = loss_t.item()
                grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                prof.step()

                total_loss  += loss
                num_batches += 1

                if (args.lr_warmup_target > 0):
                    for pg in optimizer.param_groups:
                        if (num_batches < total_batches //10):
                            pg['lr'] = args.lr + num_batches * increase_per_batch

    
                if not interactive and num_batches % heartbeat_every == 0:
                    avg = total_loss / num_batches
                    dbg_output(f"  [{num_batches}/{total_batches}] loss={avg:.4f}  grad_norm={grad_norm:.3f}")
                    getattr(transformer, '_orig_mod', transformer).dbg_output_health_check()
                    if scheduler and args.lr_schedule == 'plateau':
                        scheduler.step(avg)
                        dbg_output(f"  LR={optimizer.param_groups[0]['lr']:.6f}")

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
            if scheduler and args.lr_schedule != 'plateau':
                scheduler.step()
                dbg_output(f" LR={optimizer.param_groups[0]['lr']:.6f}")

            # End of batch inner loop
            torch.save({'model': transformer.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': i+1}, args.save_model)
            dbg_output(f"Checkpoint saved to {args.save_model}")

            if args.shared_checkpoint_dir and (i + 1) % args.shared_checkpoint_every == 0:
                import subprocess
                os.makedirs(args.shared_checkpoint_dir, exist_ok=True)
                shared_pth = os.path.join(args.shared_checkpoint_dir, 'model.pth')
                torch.save({'model': transformer.state_dict(), 'epoch': i+1}, shared_pth)
                dbg_output(f"Shared checkpoint saved → {shared_pth}")
                if args.entropy_csv:
                    subprocess.Popen([
                        sys.executable, 'analyze_checkpoints.py',
                        '--extract-single', shared_pth,
                        '--epoch', str(i + 1),
                        '--output-csv', args.entropy_csv,
                    ])

        # End of Epoch Loop


    dbg_output(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit = 20))
    dbg_output(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit = 20))
    prof.export_chrome_trace("trace.json")


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
