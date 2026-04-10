# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

import AnnUtils
import numpy as np
from debug import *
from heuristics import *

# shape has dataset size(n), input size(d), numnodes(d)

# forward pass:
# =============
# softmax will have m*n inputs and m*n outputs
# unlike sigmoid the operation requires all m*n to do the softMax, but the 
# shapes are same on the input/output. i.e both are m*n

# backprop:
# ========
# m*n is the outputError. if its the final layer, there are no
# inputError. instead the outputError becomes target - outputs
# shapes are tallying

class SoftMaxSLayer:
    nextLayer = None
    prevLayer = None

    def __init__(self, name, shape, lr):
        self.name = name
        self.shape = shape
        pass

    def setPrevLayer (self, prevLayer ):
        self.prevLayer = prevLayer

    def setNextLayer(self, nextLayer):
        self.nextLayer = nextLayer

    def setTargets(self, targets):
        
        targets = targets.flatten()
        self.targetIDs = targets
        n = targets.shape[0]
        
        one_hot = np.zeros((10,n))
        one_hot[targets, np.arange(n)] = 1
        self.targets = one_hot
 

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.outputs = AnnUtils.softMax(inputs)
        info ("softMax outputs")
        info ("===============")
        info (self.outputs)


        if (self.nextLayer != None):
            self.nextLayer.forward(self.outputs)
            debug (self.name, "input errors not relevant");
            debug (self.name, "forwarding to the next layer");
        else:
            #Final output stage, calculate the errors and keep handy
             
            self.inputError = self.outputs - self.targets 
            debug (self.name, "input errors")
            debug (self.name, "============")
            debug (self.name, self.inputError)


        p_true = self.outputs[self.targetIDs, np.arange(self.targetIDs.shape[0])]
        loss = -np.mean(np.log(p_true))

        if (heuristics.isTraining):
            heuristics.epochData["trainingLoss"] = loss 
        else:
            heuristics.finalData["validationLoss"] = loss

        info (self.name, "Current Loss")
        info ("=============================")
        info (loss)


        res1=np.argmax(self.outputs, axis=0 )
        accuracy = np.mean(res1 == self.targetIDs)

        if (heuristics.isTraining):
            heuristics.epochData["trainingAccuracy"] = accuracy
        else:
            heuristics.finalData["validationAccuracy"] = accuracy

        debug (self.name, "Current Accuracy")
        debug ("=============================")
        debug (accuracy)


        return res1
    
    def backprop(self, inputError):
        if (inputError is None and self.inputError is None):
            error("Error: backprop without valid error input")
            raise Exception("Error: backprop without valid error input")
            return

        if (inputError is None):
            inputError = self.inputError

        return self.prevLayer.backprop(inputError)
