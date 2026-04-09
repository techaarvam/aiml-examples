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
        mean = np.mean(trainingInputs, axis=1, keepdims=True)
        std = np.std(trainingInputs, axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        if (args.normalize):
            self.trainingInputs = (trainingInputs - mean) / std
        else: 
            self.trainingInputs = trainingInputs

        self.trainingTargets = np.array (digits.target[0:1000])
    
        validationInputs = np.array ( digits.data[1000:]).T 
        std = np.where(std < 1e-8, 1.0, std)
        if (args.normalize):
            self.validationInputs = (validationInputs - mean ) / std
        else: 
            self.validationInputs = validationInputs 
        self.validationTargets = np.array( digits.target[1000:])
        # the .T is to keep the math as y = W.Tx + b 
        # feel a bit more linear algebra friendly

    
