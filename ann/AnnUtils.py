# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

import numpy as np
import math
from debug import *
from argParser import *

def getSeed(text):
    chars = np.frombuffer(text.encode(), dtype=np.uint8)

    return np.sum(chars) + args.seed

def weightedAdd( W, x, b):
    info ("Shapes: W.T @ x + b")
    info (W.T.shape, x.shape, b.shape)
    info ("bias vals")
    info (b)
    return W.T @ x + b

# Keeping this commented out (instead of removing), was a nice learning on
# preveting the exp causing a blowup in softMax
# def softMax( z ):
#    exp_z = np.exp(z) 
#    return exp_z / np.sum(exp_z) 

def softMax(z):
    z_stable = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_stable)
    # this (exp_z/ sum(exp_z, axis=0)) is converting the output to probabilities
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
   

def ReLu ( z ):
    return np.where( z>0, z, 0)

# TBD: Handle the case when z is large using
# exp (z) / (1 + exp(z))

def sigmoid ( z ):
    return 1 / (1 + np.exp(-z))
