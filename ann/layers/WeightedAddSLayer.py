# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

from debug import *
from argParser import *
import re
from heuristics import *

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

        matchObj = re.match(".*([0-9]+).*", name)

        if (matchObj is None): 
            self.layerNum = "0"
        else: 
            self.layerNum = matchObj[1]

        rng = np.random.default_rng(seed=AnnUtils.getSeed(name))
        self.weights = (rng.random(size=(d, m))-args.weight_skew) * args.weight_scale
        #self.weights = np.zeros((d,m)) 

        self.bias = np.zeros((m,1))
        self.output = np.zeros((m,n))
        self.learningRate = lr
 
    def forward(self, inputs):
        # All these copies look inefficient!
        # later...

        self.inputs = inputs.copy() 
        self.outputs = AnnUtils.weightedAdd(self.weights, inputs ,self.bias)

        fracLargeZ = np.mean(self.outputs>3) 
         
        heuristics.epochData["fracLargeZ_"+self.layerNum] = fracLargeZ
        

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
        weightDirectionT = weightDirection.T
        self.weights -= weightDirectionT * lr

        weightNorm = np.linalg.norm(self.weights, axis=1)
        weightDirectionNorm = np.linalg.norm(weightDirectionT, axis=1)
        safeWeightNorm = np.where(weightNorm < 1e-12, 1.0, weightNorm)
        updateRatio = weightDirectionNorm / safeWeightNorm
        maxUR = np.max(updateRatio)
        minUR = np.min(updateRatio)
        meanUR = np.mean(updateRatio)

        heuristics.epochData["MeanWAgradientNorm_"+self.layerNum] = np.mean(weightDirectionNorm)
        heuristics.epochData["MaxWAupdateRatio_"+self.layerNum] = maxUR
        heuristics.epochData["MinWAupdateRatio_"+self.layerNum] = minUR
        heuristics.epochData["MeanWAupdateRatio_"+self.layerNum] = meanUR

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
