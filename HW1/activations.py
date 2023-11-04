from typing import Any
import numpy as np


class Sigmoid():
    def __init__(self):
        pass
        
    def __call__(self, inputs):
        """Compute Sigmoid for given inputs

        Args:
            input (array): inputs of shape (minibatchsize, num_units)
        """
        return 1 / (1 + np.exp(-inputs))
    
    def __repr__(self):
        return f'Sigmoid'

class Softmax():
    def __init__(self):
        pass

    def __call__(self, inputs):
        """Compute Softmax for given inputs

        Args:
            inputs (array): inputs of shape (minibatchsize, 10)

        Returns:
            array: _description_
        """
        return np.exp(inputs) / np.sum(np.exp(inputs), axis=1, keepdims=True)
    
    def __repr__(self):
        return f'Softmax'