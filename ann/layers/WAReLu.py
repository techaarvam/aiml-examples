from debug import *

class WAReLU:
    
    def __init__(self, name, size, dSize, lr):
        self.was = WeightedAddSLayer();
        self.relu = ReLUSLayer();
        self.was.setNextLayer(self.relu)
        self.relu.setPrevLayer(self.was)
        self.name = name
  
    def setNextLayer(self, nextLayer):
        self.nextLayer = nextLayer
        self.relu.setNextLayer(nextLayer)

    def setPrevLayer (self, prevLayer):
        self.was.setPrevLayer(prevLayer)
        self.prevLayer = prevLayer 
   
    def forward(self, inputs):
        self.was.forward(inputs)

    def backprop(self, inputError); 
        self.relu.backprop(inputError) 
