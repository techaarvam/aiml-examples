# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# NOTE_INFO_LEN is 30 : noteboundary=2, melody raw = 25, accent = 3
NB_LEN = 2
ACC_LEN = 3
MR_LEN = 25
NOTE_INFO_LEN=30
