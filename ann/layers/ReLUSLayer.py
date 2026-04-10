# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

import AnnUtils
from debug import *
from heuristics import *
import numpy as np

class ReLuSLayer:
    inputs = None
    outputs = None
    prevLayer = None
    nextLayer = None

    def __init__(self, name, shape, lr):
        self.name=name

    def setPrevLayer (self, prevLayer ):
        self.prevLayer = prevLayer

    def setNextLayer(self, nextLayer):
        self.nextLayer = nextLayer


    def forward (self, inputs):
        self.outputs = AnnUtils.ReLu(inputs)
        self.inputs = inputs.copy()

        heuristics.epochData["DeadReLUFrac"] = np.mean(self.outputs == 0)

        return self.nextLayer.forward (self.outputs)

    def backprop(self, inputError):
        outputError = inputError * np.where(self.inputs > 0, 1, 0)
        return self.prevLayer.backprop(outputError)
