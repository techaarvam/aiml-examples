# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
import DataInput
import numpy as np
import torch
import musicRNN
from argParser import *
from debug import *
import common
from seeder import *
from torch import nn
from torch.utils.data import DataLoader
import random


set_seed(args.seed)
set_verbosity(args.verbosity)

d = DataInput.DataInput()

rnn = musicRNN.musicRNN(common.NOTE_INFO_LEN, args.hidden_dim).to(common.device)

if args.model_file:
    rnn.load_state_dict(torch.load(args.model_file, map_location=common.device))
    dbg_output(f"Loaded model from {args.model_file}")
else:
    # sum up the 3 loss functions!
    loss_fn1 = nn.CrossEntropyLoss()
    loss_fn2 = nn.CrossEntropyLoss()
    loss_fn3 = nn.CrossEntropyLoss()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params=rnn.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(params=rnn.parameters(), lr=args.lr)

    # Lets not do the random_split for this case.
    # Lets make the next word generation interactive as a chat bot!. No validation error measurement then!

    train_loader = DataLoader(d, batch_size=args.batch_size)


    for i in range(0, args.epochs):
        total_loss = 0.0
        num_batches = 0

        for inputs, labels in train_loader:
            dInputs, dLabels = inputs.to(common.device), labels.to(common.device)
            optimizer.zero_grad()

            # the unsqueeze(1) because window_size is the dim=1, which is
            # 1 in the MusicRnn, but was >1 in langRNN
 
            output, h = rnn.forward(dInputs)

            mask  = dLabels[:,:,0].bool()   # True at note boundary positions

            loss1 = loss_fn1(output[:,:,0:common.NB_LEN].reshape(-1,common.NB_LEN), dLabels[:,:,0].reshape(-1)) #NoteBoundary — all timesteps
            loss2 = loss_fn2(output[:,:,common.NB_LEN:common.NB_LEN+common.MR_LEN][mask], dLabels[:,:,1][mask]) #MelodyRaw — boundary only
            loss3 = loss_fn3(output[:,:,common.NB_LEN+common.MR_LEN:][mask],          dLabels[:,:,2][mask])     #Accent — boundary only
 
            loss = loss1 + loss2 + loss3

            loss.backward()
            # Hack?!
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        nb_preds = output[:,:,0:common.NB_LEN].argmax(dim=-1).float().mean().item()
        dbg_output(f"Epoch {i+1}: loss={total_loss/num_batches:.4f}  boundary_rate={nb_preds:.3f}")

    torch.save(rnn.state_dict(), args.input + ".model.pt")
    dbg_output(f"Model saved to {args.input}.model.pt")

# All code above this point are man-made!
# All code below this line are AI-made!
# Taking the short-cut on the inference side.

# ── BEGIN AI-CODED SECTION (Claude Sonnet 4.6) ────────────────────────────────
# Inference: generate args.output_size samples autoregressively, validate each,
# save accepted samples to generated.npz, print acceptance rate.

from validate import validate_sample

rnn.eval()
accepted = []
total    = args.num_samples
seq_len  = args.seq_len

for sample_idx in range(total):
    melody_raws = []
    accents     = []
    boundaries  = []

    start_idx  = random.randint(0, len(d) - 1)
    start_seq  = d.rawData[start_idx].unsqueeze(0).to(common.device)
    warmup_len = random.randint(1, d.seq_len // 2)
    with torch.no_grad():
        _, h = rnn.forward(start_seq[:, :warmup_len, :])
    x       = start_seq[:, warmup_len:warmup_len + 1, :]
    prev_mr = 0
    prev_ac = 0

    for t in range(seq_len):
        with torch.no_grad():
            out, h = rnn.forward(x, h)

        nb_pred = out[:, 0, 0:common.NB_LEN].argmax(dim=-1).item()
        mr_pred = out[:, 0, common.NB_LEN:common.NB_LEN+common.MR_LEN].argmax(dim=-1).item()
        ac_pred = out[:, 0, common.NB_LEN+common.MR_LEN:].argmax(dim=-1).item()

        # enforce hold rule — no change between boundaries
        if not nb_pred:
            mr_pred = prev_mr
            ac_pred = prev_ac

        melody_raws.append(mr_pred - 12)
        accents.append(ac_pred)
        boundaries.append(bool(nb_pred))

        prev_mr = mr_pred
        prev_ac = ac_pred

        next_x = torch.zeros(1, 1, common.NOTE_INFO_LEN).to(common.device)
        next_x[0, 0, nb_pred]                                  = 1.0
        next_x[0, 0, common.NB_LEN + mr_pred]                 = 1.0
        next_x[0, 0, common.NB_LEN + common.MR_LEN + ac_pred] = 1.0
        x = next_x

    mr_arr = np.array(melody_raws)
    ac_arr = np.array(accents)
    nb_arr = np.array(boundaries)

    if validate_sample(mr_arr, ac_arr, nb_arr):
        accepted.append((mr_arr, ac_arr, nb_arr))
    elif sample_idx < 5:
        print(f"Sample {sample_idx}: nb[0]={nb_arr[0]}, mr unique={np.unique(mr_arr)}, nb_rate={nb_arr.mean():.2f}")

dbg_output(f"Validation: {len(accepted)}/{total} accepted ({100*len(accepted)/total:.1f}%)")

if accepted:
    np.savez('generated.npz',
             melody_raw    = np.stack([s[0] for s in accepted]),
             accent        = np.stack([s[1] for s in accepted]),
             note_boundary = np.stack([s[2] for s in accepted]))
    dbg_output(f"Saved {len(accepted)} samples → generated.npz")

# ── END AI-CODED SECTION ──────────────────────────────────────────────────────




