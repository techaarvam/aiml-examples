from debug import *
from argParser import *

# shape has dataset size(n), input size(d), numnodes(m) 

# forward pass shapes:
#   weights m*d input d*n output m*n

# backprop shapes:
#   inputErrors(grads) m*n 
#   outputErrors(grads) d*n

#    W.T is d*m @ m*n gives d*n 
#   Satisfying to see this tally!

import numpy as np
import AnnUtils

class WeightedAddSLayer:
    prevLayer = None

    def setPrevLayer (self, prevLayer ):
        self.prevLayer = prevLayer

    def setNextLayer(self, nextLayer):
        self.nextLayer = nextLayer


    def __init__(self, name, shape, lr):
        (n,d,m) = shape
        self.name = name
        rng = np.random.default_rng(seed=args.seed)
        self.weights = (rng.random(size=(d, m))-0.5) * args.weight_scale
        #self.weights = np.zeros((d,m)) 
        self.bias = np.zeros((m,1))
        self.output = np.zeros((m,n))
        self.learningRate = lr
 
    def forward(self, inputs):
        # All these copies look inefficient!
        # later...

        self.inputs = inputs.copy() 
        self.outputs = AnnUtils.weightedAdd(self.weights, inputs ,self.bias)

        if (self.nextLayer == None): 
            raise Exception("Error: WeightedAdd Layer: forward pass attempt \
                               on an incomplete topology");
        return self.nextLayer.forward(self.outputs)

 
    def backprop(self, inputError):
        weightDirection = (1/inputError.shape[1]) * (inputError @ self.inputs.T)
        info ("shape of inputError is:", inputError.shape)
        
        biasDirection = (1/inputError.shape[1]) * np.sum(inputError, axis=1, keepdims=True)
        lr = self.learningRate.getLR()
        biasLr = self.learningRate.getBiasLR()

        info ("Shapes for weights.T and inputError below")
        info (self.weights.T.shape, inputError.shape)
        outputError = self.weights @ inputError
        
        # Weight update must happen after the outputError(grad) computation
        self.weights -= weightDirection.T * lr

        # Bias may benefit from its own learning rate
        info (self.name, " biasDirection ")
        info ("===============")
        info (biasDirection)
        debug (self.name, " bias ")
        debug ("===============")
        debug (self.name, self.bias)
        debug  (" inputError ")
        debug ("===============")
        debug (self.name, inputError)

        self.bias -= biasDirection * biasLr

        info(self.name, " biasLr ")
        info ("===============")
        info (biasLr)
        debug (self.name, " weights ")
        debug (self.name, "===============")
        debug (self.weights)

        if (self.prevLayer is not None):
            return self.prevLayer.backprop(outputError)
        else:
            return outputError


