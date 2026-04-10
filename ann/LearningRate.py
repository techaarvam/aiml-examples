# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

from argParser import *
# place holder for further enhancing for different 
# adaptive learning rate algorithms. 

class LearningRate:
    TYPE1=1
    TYPE2=2
    def __init__(self, variant):
        if (variant == self.TYPE1):
            self.lr=args.lr1
            self.biasLR=args.lr1
        else:
            self.lr=args.lr2
            self.biasLR=args.lr2
 
    def getLR(self):
        return self.lr

    def getBiasLR(self):
        return self.biasLR
