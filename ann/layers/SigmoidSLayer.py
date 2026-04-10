# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

import AnnUtils
from debug import *
from heuristics import *
import numpy as np

#shape has dataset size(n), input size(d), numnodes(m) 

#forward:
#input will be m*n
#output will also be m*n

#backprop:
# inputError will be m*n
# outputError will be m*n

# From the above looks like dimensions are all the same
# so can directly operate with them. i.e all operations will be preserving
# dimensions like sigmoid, hadamard product and such. Tallies - tick.


class SigmoidSLayer:
    inputs = None
    outputs = None 
    prevLayer = None
    nextLayer = None

    def __init__ (self, name, shape, lr):
        self.name = name
 
    def setPrevLayer (self, prevLayer ):
        self.prevLayer = prevLayer

    def setNextLayer(self, nextLayer):
        self.nextLayer = nextLayer


    def forward(self, inputs):
        self.outputs = AnnUtils.sigmoid(inputs)
        debug ("Sigmoid saturation checks")
        debug ("==========================")
        debug (self.name, (np.min(inputs), np.max(inputs), np.mean(inputs)))
        debug (self.name, (np.min(self.outputs), np.max(self.outputs), np.mean(self.outputs)))
        debug (self.name, "low sigmoid frac:", np.mean(self.outputs < 0.01))
        debug (self.name, "high sigmoid frac:", np.mean(self.outputs > 0.99))
        debug (self.name, "saturating sigmoid frac:", 
            np.mean((self.outputs < 0.01) | (self.outputs > 0.99)))
        debug ("==========================")

        debug ("Sigmoid derivative is", np.mean(self.outputs * (1 - self.outputs)))
        debug ("==========================")

        heuristics.epochData["sigmoidSaturationFrac"] = \
            np.mean((self.outputs < 0.01) | (self.outputs > 0.99))

        heuristics.epochData["sigmoidDerivativeAvg"] = \
            np.mean(self.outputs * (1 - self.outputs))

        self.inputs = inputs.copy()
        return self.nextLayer.forward(self.outputs)

    def backprop(self, inputError):
        # hadamard product 
        outputError = inputError * self.outputs * (1 - self.outputs)
        return self.prevLayer.backprop(outputError)
