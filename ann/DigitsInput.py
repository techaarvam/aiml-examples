# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

from sklearn import datasets
import numpy as np

from argParser import *

class DigitsInput:
    # Keeping these variables here for 
    # having one place to see the data-members 

    trainingInputs = None
    trainingTargets = None

    validationInputs = None
    validationTargets = None

    def __init__(self):

        digits = datasets.load_digits()
 

        trainingInputs = np.array  (digits.data[0:1000]).T 
       
        # Using axis=0, i.e persample mean of the 64 features,
        # we will get 1000 means due to the 0:1000
        mean = np.mean(trainingInputs, axis=0, keepdims=True)
        std = np.std(trainingInputs, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        if (args.normalize):
            self.trainingInputs = (trainingInputs - mean) / std
        else: 
            self.trainingInputs = trainingInputs

        self.trainingTargets = np.array (digits.target[0:1000])
    
        validationInputs = np.array ( digits.data[1000:]).T 
        validationMean = np.mean(validationInputs, axis=0, keepdims=True)
        validationStd = np.std(validationInputs, axis=0, keepdims=True)
        validationStd = np.where(validationStd < 1e-8, 1.0, validationStd)
        if (args.normalize):
            self.validationInputs = (validationInputs - validationMean) / validationStd
        else: 
            self.validationInputs = validationInputs 
        self.validationTargets = np.array( digits.target[1000:])
        # the .T is to keep the math as y = W.Tx + b 
        # feel a bit more linear algebra friendly

    
