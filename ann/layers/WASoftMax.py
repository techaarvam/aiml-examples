# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

from debug import *

# shape has dataset size(n), input size(d), numnodes(d)

# forward pass:
# ============
# similar to the WASigmoid the shapes are the following
#  weight m*d, input d*n, bias m*n, output m*n

# backprop:
# ========
# inputError is m*n (or may not exist as the final layer)
# outputError for softMax stays as m*n
# outputError for the WA will be d*n (shared code and it matches)

from layers.WeightedAddSLayer import *
from layers.SoftMaxSLayer import *

class WASoftMax:
    
    def __init__(self, name, shape,  lr):
        self.name = name
        self.weightedAdd = WeightedAddSLayer(name, shape, lr)
        self.softMax = SoftMaxSLayer(name, shape,lr)
        self.weightedAdd.setNextLayer(self.softMax)
        self.softMax.setPrevLayer(self.weightedAdd)

    def setTargets(self, targets):
        self.softMax.setTargets(targets)
  
    def setNextLayer(self, nextLayer):
        self.nextLayer = nextLayer
        self.softMax.setNextLayer(nextLayer)

    def setPrevLayer (self, prevLayer):
        self.weightedAdd.setPrevLayer(prevLayer)
        self.prevLayer = prevLayer 
   
    def forward(self, inputs):
        return self.weightedAdd.forward(inputs)

    def backprop(self, inputError=None): 
        # inputError will be None if WASoftMax is the final layer
        self.softMax.backprop(inputError) 
