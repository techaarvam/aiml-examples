# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

from debug import *
from layers.WeightedAddSLayer import *
from layers.ReLUSLayer import *

class WAReLU:
    
    def __init__(self, name, shape, lr):
        self.shape = shape
        self.name = name
 
        info ("shape: ", shape)

        self.weightedAdd = WeightedAddSLayer(name, shape, lr);
        self.relu = ReLuSLayer(name, shape, lr);

        self.weightedAdd.setNextLayer(self.relu)
        self.relu.setPrevLayer(self.weightedAdd)
 
    def setTargets ( self, targets):
        self.relu.setTargets(targets)
 
    def setNextLayer(self, nextLayer):
        self.nextLayer = nextLayer
        self.relu.setNextLayer(nextLayer)

    def setPrevLayer (self, prevLayer):
        self.weightedAdd.setPrevLayer(prevLayer)
        self.prevLayer = prevLayer 
   
    def forward(self, inputs):
        return self.weightedAdd.forward(inputs)

    def backprop(self, inputError): 
        return self.relu.backprop(inputError) 
