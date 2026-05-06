# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
from argParser import *
from debug import *
import numpy as np
import torch
import common

class DataInput():
    def __init__(self):
        
        music = np.load(args.input)
        noteBoundary = torch.tensor(music['note_boundary'])
        melodyRaw = torch.tensor(music['melody_raw']) + 12 # get to 0-24
        accent = torch.tensor(music['accent'])

        num_samples = accent.shape[0]
        seq_len = accent.shape[1]

        self.num_samples = num_samples
        self.seq_len = seq_len



        self.rawData = torch.zeros(num_samples, seq_len, common.NOTE_INFO_LEN)

        self.rawData.scatter_ (2, noteBoundary.long().unsqueeze(-1), 1.0)
        
        mrShifted = (melodyRaw + common.NB_LEN).unsqueeze(-1)
        self.rawData.scatter_(2, mrShifted, 1.0)

        accentShifted = (accent + common.NB_LEN + common.MR_LEN).unsqueeze(-1)
        self.rawData.scatter_(2, accentShifted, 1.0)

        # nn.CrossEntropyLoss takes integer targets , not one-hot. 
        # the output logits are softMax'd internally by CrossEntropy
        # nuances!
        self.targets = torch.stack ( [noteBoundary.long(), melodyRaw, accent ], dim=2 )

 
        debug (f"number of samples in the training data: {num_samples}. Sequence length = {seq_len} ")


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        data  = self.rawData[index, :-1]    # (seq_len-1, 30)
        label = self.targets[index, 1:]     # (seq_len-1, 3)
        return (data, label)


if __name__ == "__main__":
    set_verbosity(DEBUG)
    d = DataInput()


