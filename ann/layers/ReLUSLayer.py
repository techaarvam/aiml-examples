import AnnUtils
from debug import *

class ReLuSLayer:
    inputs = None
    outputs = None
    prevLayer = None
    nextLayer = None

    def setPrevLayer (self, prevLayer ):
        self.prevLayer = prevLayer

    def setNextLayer(self, nextLayer):
        self.nextLayer = nextLayer


    def forward (self, inputs):
        self.outputs = AnnUtils.ReLU(inputs)
        self.inputs = inputs.copy()
        self.nextLayer.forward (self.outputs)
        # return self.outputs

    def backprop(self, inputError):
        outputError = inputError * np.where( inputs > 0, 1,0)
        self.prevLayer.backprop (self.outputError)
        #return outputError

