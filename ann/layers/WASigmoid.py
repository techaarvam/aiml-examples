from debug import *

# shape has dataset size(n), input size(d), numnodes(m) 

# output will be m*n
# weights are m*d
# input will be d*n 
# bias will be also m*n (each node has its bias)

# inputError m*n
# outputError from sigmoid is m*n
# outputError from WeightedAdd is d*n, so outputError from weightedAddigmoid is d*n 

from layers.WeightedAddSLayer import *
from layers.SigmoidSLayer import *

class WASigmoid:
    
    def __init__(self, name, shape, lr):
        self.shape = shape
        self.name = name

        info ("shape: ", shape)
        self.weightedAdd = WeightedAddSLayer(name, shape, lr)

        self.sigmoid = SigmoidSLayer(name, shape, lr)

        self.weightedAdd.setNextLayer(self.sigmoid)
        self.sigmoid.setPrevLayer(self.weightedAdd)
  

    def setTargets(self, tarets):
        self.sigmoid.setTargets(targets)
  
    def setNextLayer(self, nextLayer):
        self.nextLayer = nextLayer
        self.sigmoid.setNextLayer(nextLayer)

    def setPrevLayer (self, prevLayer):
        self.weightedAdd.setPrevLayer(prevLayer)
        self.prevLayer = prevLayer 
   
    def forward(self, inputs):
        return self.weightedAdd.forward(inputs)

    def backprop(self, inputError): 
        return self.sigmoid.backprop(inputError) 
