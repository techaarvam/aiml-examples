# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

import AnnUtils
from layers.WASigmoid import *
from layers.WAReLu import *
from layers.WASoftMax import *
from DigitsInput import *
from LearningRate import *
from argParser import *
from heuristics import *

import debug
debug.VERBOSITY=args.verbosity

from debug import *

# Activation Type constants
SIGMOID=1
RELU=2

def main():
   
   dataset = DigitsInput() 
   lr1 = LearningRate(LearningRate.TYPE1)
   lr2 = LearningRate(LearningRate.TYPE2)

   #shape ( n-dataset size, d-inputsize, m-numnodes)
   if (args.actType == SIGMOID):
       layer1 = WASigmoid("LAYER1: WASigmoid", (1000, 64, args.hidden_size), lr1) 
   else:
       layer1 = WAReLU("LAYER1: WAReLU", (1000, 64, args.hidden_size), lr1) 


   layer2 = WASoftMax("LAYER2: WASoftMax", (1000, args.hidden_size, 10), lr2)

   layer2.setPrevLayer(layer1)
   layer2.setTargets(dataset.trainingTargets)
   layer1.setNextLayer(layer2)
   heuristics.dumpConfig()

   # trainining loop of 1000(epochs default) iterations
   for i in range(0,args.epochs):
       # in the current impl, state is actually maintained internally
       # there isnt a need to get outputs and pass to backprop

       # TBD: Check if the shape is correct and whether it needs a transpose
       info ("Shape of the trainingInputs: ", dataset.trainingInputs.shape)
       layer1.forward(dataset.trainingInputs)
       layer2.backprop() 
       if (i% args.report_every == 0):
           heuristics.dumpEpoch()

   heuristics.isTraining = False
   # Lets validate

   layer2.setTargets(dataset.validationTargets)
   output = layer1.forward(dataset.validationInputs)

   info ("output shape is", output.shape)
   for i in range(0, len(dataset.validationTargets)):
       debug ("output for the current input is: ", output[i])
       debug ("target for the current input is: ", dataset.validationTargets[i])
   heuristics.dumpFinal()


if __name__ == "__main__":
    main()
